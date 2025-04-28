import numpy as np
import torch
import torch.nn.functional as F

from utilis import npTorchCGPU_manula_seed
from architect import Model

class expert_bandit_conformal:
    def __init__(self, head_layer, output_dim, hyperParamDict, device):
        self.T, self.gam, self.K = hyperParamDict["T"], hyperParamDict["gam"], output_dim
        self.bpolicy = hyperParamDict["pname"]
        self.thresh = hyperParamDict["threshInit"] * torch.ones([1, self.K]).to(device)

        self.epoch = hyperParamDict["epoch"]
        self.tauDecay = hyperParamDict["tauDecay"]
        self.mt, self.vt = 0., 0.
        self.taulr = hyperParamDict["taulr"]
        self.adp_eta2 = hyperParamDict["adp_eta2"]
        # self.smoothPara = hyperParamDict["smoothPara"]

        self.CElossnormDelta = hyperParamDict["CElossnormDelta"]
        self.CKlossnormDelta = hyperParamDict["CKlossnormDelta"]
        self.scaleCK_val_pull = hyperParamDict["scaleCK_val_pull"]
        self.checklossfun = hyperParamDict["checklossfun"]
        self.tau_cosAnl = hyperParamDict["tau_cosAnl"]
        self.tau_scale = 1.

        self.taulrJump = hyperParamDict["taulrJump"]
        if self.tau_cosAnl + (len(self.taulrJump) > 0) > 1:
            raise ValueError("tau_cosAnl, and taulrJump cannot be true at the same time")
        # self.tauGD = hyperParamDict["tauGD"]

        # self.factor = hyperParamDict["factor"]
        self.kreg = hyperParamDict["kreg"]
        self.score_amp = hyperParamDict["score_amp"]

        ######################
        self.sigma = hyperParamDict["sigma"]
        self.eta_w = hyperParamDict["eta_w"]
        self.adp_etaW = hyperParamDict["adp_etaW"]
        self.eta2 = torch.tensor(hyperParamDict["eta2"], device=device).reshape(-1, 1)

        self.num_expert = len(hyperParamDict["eta2"])
        self.w = torch.ones([self.num_expert, self.K], device=device)
        self.thresh_dist = self.w / torch.sum(self.w, 0)

        if type(hyperParamDict["expertThresh_init"]) != "str":
            self.thresh_tensor = torch.ones([self.num_expert, self.K], device=device) * hyperParamDict["expertThresh_init"]
        elif hyperParamDict["expertThresh_init"] == "uniform":
            self.thresh_tensor = torch.rand([self.num_expert, self.K], device=device)

        # self.result["holdout_loss"] = np.zeros(self.epoch)

        print(f"thresh_tensor = {self.thresh_tensor}")
        ######################

        self.quant_opt, self.quant_para = list(hyperParamDict["quantOpt"].items())[0]
        self.learning_rate, myweight_decay, mybetas = hyperParamDict["learning_rate"], hyperParamDict["weight_decay"], hyperParamDict["betas"]

        self.device = device
        self.seed = hyperParamDict["seed"]
        self.lrSched, self.lrSchedPara = list(hyperParamDict["lrSched"].items())[0]
        self.conformal_score = hyperParamDict['conformal_score']

        npTorchCGPU_manula_seed(self.device, self.seed)
        self.modelLogit = Model(hyperParamDict["useBackbone"], head_layer, output_dim).to(device)
        self.optimizer = torch.optim.Adam(self.modelLogit.parameters(), lr=self.learning_rate, weight_decay=myweight_decay, betas=mybetas)
        if self.lrSched == "CosineAnnealingLR":
            self.myscheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.epoch, eta_min=self.lrSchedPara)
        elif self.lrSched == "MultiStep":
            self.myscheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.lrSchedPara)

        self.result = {
                       "objtr_batch": np.zeros(1 + self.T),
                       "accum_count": np.zeros([1 + self.T, self.K]),
                       "accum_recall": np.zeros([1 + self.T, self.K]),
                       "accum_card": np.zeros([1 + self.T, 3]),
                       "threshold": np.zeros([1 + self.T, self.K]),

                       "holdout_recall": np.zeros([self.epoch, self.K]),
                       "holdout_card": np.zeros([self.epoch, 2]),
                       "holdout_loss": np.zeros(self.epoch),
                       "lr_arr": np.zeros(self.epoch)}

        for key, val in hyperParamDict.items():
            print(f"{key}: {val}")


    def train(self, trainLoader, **kwargs):
        if "augLoader" in kwargs:
            self.result["holdoutAug_recall"] = np.zeros([self.epoch, self.K])
            self.result["holdoutAug_card"] = np.zeros(self.epoch)

        npTorchCGPU_manula_seed(self.device, self.seed)

        t = 0
        for epoch in range(self.epoch):
            if epoch in self.taulrJump:
                self.eta2 = self.eta2 * 0.1

            for batch, (total_X, total_Y) in enumerate(trainLoader):
                if self.adp_etaW:
                    self.eta_w = (1 + t) ** (-0.5)

                total_X, total_Y = total_X.to(self.device), total_Y.to(self.device)

                self.modelLogit.train()
                self.optimizer.zero_grad()

                logits = self.modelLogit(total_X).clip(min=-15, max=15)

                prob = logits.softmax(1)
                score = self.class_Score(prob.detach())
                self.perfSummary(t, total_Y, score) # set-valued prediction

                deltat, deltat_normalize, num_valid_pull = self.feedback(prob.detach(), total_Y) # estimate label
                deltat_div_pull = deltat / (num_valid_pull + 1e-06)
                logit_obj = self.bandit_crossEntropy(deltat_div_pull, prob)

                logit_obj.backward()
                self.optimizer.step()

                self.result["objtr_batch"][t] = logit_obj.item()
                t += 1
                self.threshold_update(deltat_div_pull, score, t)

            self.logResult(t, epoch, kwargs["evalLoader"], augLoader= kwargs["augLoader"] if "augLoader" in kwargs else None)

            if self.lrSched:
                self.myscheduler.step()

            if self.tau_cosAnl:
                self.tau_scale = self.lrdecay2(epoch, self.epoch)

    def lrdecay2(self, curepoch, epoch, mincoeff=1e-05, decaylength=0.9):
        decayintval = decaylength * epoch
        val = 0.5 * (1 - mincoeff) * (np.cos(np.pi / decayintval * (curepoch + 1)) + 1)
        return val * (curepoch + 1 <= decayintval) + mincoeff

    def bandit_crossEntropy(self, deltat_normalize, prob):
        return -torch.sum(deltat_normalize * torch.log(prob))

    def weighted_check_loss(self, deltat_normalize, score):
        expert_loss = torch.zeros_like(self.thresh_tensor)

        for expert, ele in enumerate(self.thresh_tensor):
            diff = score - ele.reshape(1, -1)
            expert_loss[expert] = torch.sum(deltat_normalize * diff * (self.gam - (diff < 0) * 1.), 0)

        return expert_loss

    def class_Score(self, prob, rankCost=0.01):
        if self.conformal_score == "softmax":
            return prob * self.score_amp

        score_sort, score_rank = torch.sort(prob, 1, descending=True)
        rank = score_rank.argsort(1)
        newscore = score_sort.cumsum(1).gather(1, rank)

        APS = newscore - prob * torch.rand(prob.shape).to(self.device)

        if self.conformal_score == "APS":
            return (1 - APS) * self.score_amp
        elif self.conformal_score == "RAPS":
            RAPS = APS + torch.fmax(rankCost * (rank + 1 - self.kreg), torch.zeros(1).to(self.device))
            return (1 - RAPS) * self.score_amp

    def feedback(self, policyMat, Yt):
        if self.bpolicy == "full_feedback":
            return F.one_hot(Yt, self.K).float(), None, len(Yt)

        if self.bpolicy == "uniform":
            policyMat = torch.ones_like(policyMat) / self.K

        At = torch.multinomial(policyMat, 1).squeeze()
        num_valid_pull = 0

        deltat = torch.zeros_like(policyMat)
        for i, at in enumerate(At):
            if at == Yt[i]:
                deltat[i, at] = 1 / policyMat[i, at]
                num_valid_pull += 1


        sumDeltat0 = deltat.sum(0)
        deltat_normalize = deltat / (sumDeltat0 + (sumDeltat0 == 0.) * 1.)

        return deltat, deltat_normalize, num_valid_pull



    def w_update(self, deltat_normalize, score):
        check_loss = self.weighted_check_loss(deltat_normalize, score)
        self.w = self.w * torch.exp(-self.eta_w * check_loss)
        self.thresh_dist = self.w / torch.sum(self.w, 0, keepdim=True)

    def threshold_update(self, deltat_normalize, score, t):
        self.w_update(deltat_normalize, score) # cannot swap w_update with gradient updating
        gradient = torch.zeros_like(self.thresh_tensor)

        for expert, thresh in enumerate(self.thresh_tensor):
            thresh = thresh.reshape(1, -1)
            gradient[expert] = -torch.sum(deltat_normalize * (self.gam - (score < thresh) * 1.), 0) + self.tauDecay * self.thresh

        if self.quant_opt == "SGD":
            self.mySGD(gradient)
        elif self.quant_opt == "SGDM":
            self.mySGDM(gradient, self.quant_para)
        elif self.quant_opt == "ADAM":
            self.myADAM(gradient, t, self.quant_para)
        else:
            raise ValueError("optimizer is not specified")

    def _setPred(self, scoret):
        phit = [None for _ in scoret]
        for i, score in enumerate(scoret):
            phit[i] = set([k for k, ele in enumerate(score) if k > 0 and ele >= 0])

            if not phit[i]:
                phit[i].add(0)

        return phit

    def perfSummary(self, t, Yt, score):
        self.thresh = torch.sum(self.thresh_tensor * self.thresh_dist, 0, keepdim=True) # num_expert * K,  num_expert * K
        scoret = score - self.thresh
        phit = self._setPred(scoret)

        self.result["accum_card"][t + 1] = self.result["accum_card"][t]
        self.result["accum_recall"][t + 1] = self.result["accum_recall"][t]
        self.result["accum_count"][t + 1] = self.result["accum_count"][t]
        self.result["threshold"][t + 1] = self.thresh.to("cpu").numpy()

        for idx, phi in enumerate(phit):
            phi_size = 0 if 0 in phi else len(phi)
            self.result["accum_card"][t + 1, 0] += phi_size ## overall card
            self.result["accum_count"][t + 1, Yt[idx].item()] += 1  ## count for OOD + ID

            if Yt[idx].item() > 0:
                self.result["accum_card"][t + 1, 1] += phi_size
                self.result["accum_card"][t + 1, 2] += 1

            if Yt[idx].item() in phi:
                self.result["accum_recall"][t + 1, Yt[idx].item()] += 1


    def mySGD(self, gradient):
        self.thresh_tensor = self.thresh_tensor - self.eta2 * self.tau_scale * gradient

    def mySGDM(self, gradient, beta=0.9):
        self.mt = beta * self.mt + (1 - beta) * gradient
        self.thresh_tensor = self.thresh_tensor - self.eta2 * self.tau_scale * self.mt

    def myADAM(self, gradient, t, betas=(0.9, 0.999), eps=1e-08):
        self.mt = betas[0] * self.mt + (1 - betas[0]) * gradient
        self.vt = betas[1] * self.vt + (1 - betas[1]) * gradient ** 2

        mt_hat = self.mt / (1 - betas[0] ** t)
        vt_hat = self.vt / (1 - betas[1] ** t)

        self.thresh_tensor = self.thresh_tensor - self.eta2 * self.tau_scale * mt_hat / (vt_hat ** 0.5 + eps)

    def test(self, evalLoader):
        classNum = [0] * self.K
        classRecall, card, condCard = [0] * self.K, 0, 0

        self.modelLogit.eval()
        with torch.no_grad():
            hdloss = 0.
            for batchX, trueY in evalLoader:
                batchX, trueY = batchX.to(self.device), trueY.to(self.device)
                logits = self.modelLogit(batchX)
                hdloss += F.cross_entropy(logits, F.one_hot(trueY, self.K).float())
                batchScore_test = self.class_Score(logits.softmax(1)) - self.thresh

                batchPred_test = 1. * (batchScore_test >= 0)
                pred_OOD = 1. * (batchPred_test[:, 1:].sum(1) == 0)
                batchPred_test[:, 0] = pred_OOD

                for k in range(self.K):
                    classNum[k] += torch.sum(trueY == k).item()
                    classRecall[k] += torch.sum(batchPred_test[trueY == k, k]).item()

                condCard += batchPred_test[trueY > 0].sum()
                card += batchPred_test.sum()

        total_num = sum(classNum)
        condCard = condCard.item() / (total_num - classNum[0])
        card = card.item() / total_num

        for k in range(self.K):
            classRecall[k] = classRecall[k] / classNum[k]

        return classRecall, condCard, card, hdloss / len(evalLoader)
    def logResult(self, t, epoch, evalLoader, **kwargs):
        self.result["lr_arr"][epoch] = self.optimizer.param_groups[0]['lr']
        print(f"Timestep {t}; process {int(t/self.T * 100)}%, lr = {self.result['lr_arr'][epoch]}")

        # self.result["objtr_record"][epoch] = logit_loss_avg
        self.result["holdout_recall"][epoch], condCard, card, self.result["holdout_loss"][epoch] = self.test(evalLoader)
        self.result["holdout_card"][epoch] = card, condCard
        print(f"--: holdout_card: {self.result['holdout_card'][epoch]}, holdout_recall: {self.result['holdout_recall'][epoch]}\n--: quantile: {self.thresh}")

        # if kwargs["augLoader"] is not None:
        #     self.result["holdoutAug_recall"][epoch], self.result["holdoutAug_card"][epoch] = self.test(kwargs["augLoader"])
        #     print(f"**: Augmented data cards: {self.result['holdoutAug_card'][epoch]}, Recalls: {self.result['holdoutAug_recall'][epoch]}")


