import random
import numpy as np
from typing import List


def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(x, hi))


def safe_mean(arr: np.ndarray | List[float]) -> float:
    arr = np.array(arr)
    return arr.mean() if arr.size else np.nan


def apply_label_bias(group: str, score: float, bias: float) -> float:
    # bias increases score for group A, decreases for group B
    return score * (1 + bias) if group == "A" else score * (1 - bias)


class Applicant:
    def __init__(self, uid: int, group: str, lbl_bias: float):
        self.id, self.group, self.lbl_bias = uid, group, lbl_bias
        if group == "A":
            self.has_job = random.random() < 0.9
            self.has_car = random.random() < 0.8
            self.has_house = random.random() < 0.6
            self.wealth = 50 + random.randint(0, 39)
            self.credit_score = 650 + random.randint(0, 149)
            self.education = random.randint(0, 4)
            self.loan_hist = random.randint(0, 2)
            self.risk_tol = clamp(random.gauss(0.4, 0.1))
        else:  # group == "B"
            self.has_job = random.random() < 0.7
            self.has_car = random.random() < 0.5
            self.has_house = random.random() < 0.3
            self.wealth = 30 + random.randint(0, 29)
            self.credit_score = 550 + random.randint(0, 149)
            self.education = random.randint(0, 3)
            self.loan_hist = random.randint(0, 1)
            self.risk_tol = clamp(random.gauss(0.6, 0.2))

        # common attributes
        self.num_children = random.randint(0, 4)
        self.loan_amount = 10_000 + random.randint(0, 89_999)
        self.loan_purpose = random.randint(
            1, 5
        )  # e.g., 1=house, 2=car, 3=education, 4=job, 5=other

        # derived attributes
        self.trust = np.clip(
            0.6 * self.education
            + 0.2 * np.log1p(self.wealth) / 10
            + 0.2 * np.random.uniform(0, 1),
            0,
            1,
        )
        self.fin_lit = np.clip(
            0.1 * self.education + 0.3 * self.trust + 0.3 * np.log1p(self.wealth) / 5,
            0,
            1,
        )

        self.waiting_time = 0
        self.qualified = False
        self.loan_approved = False
        self.processed = False

    def qualification_score(self) -> float:
        income_factor = 20 * self.has_job + 0.1 * self.wealth
        asset_factor = 10 * self.has_car + 15 * self.has_house
        credit_factor = (self.credit_score - 300) / 18
        education_factor = self.education * 5
        loan_history_factor = min(self.loan_hist, 5) * 2  # cap loan history effect
        dependents_penalty = -self.num_children * 2

        # loan-specific adjustments
        loan_amount_penalty = -0.05 * (self.loan_amount / max(1, self.wealth))
        loan_purpose_bonus = {
            1: 5,  # House
            2: 3,  # Car
            3: 7,  # Education
            4: 4,  # Job
            5: 1,  # Other
        }.get(self.loan_purpose, 0)

        risk_penalty = -5 * (1 - self.risk_tol)

        score = (
            income_factor
            + asset_factor
            + credit_factor
            + education_factor
            + loan_history_factor
            + dependents_penalty
            + loan_amount_penalty
            + loan_purpose_bonus
            + risk_penalty
        )

        return apply_label_bias(self.group, score, self.lbl_bias)

    def apply_decision_effect(self):
        if self.processed:
            return
        if self.loan_approved:
            # positive effects (e.g., wealth gain, asset acquisition)
            gain = 0.05 * self.loan_amount * self.fin_lit
            self.wealth += gain
            if self.loan_purpose == 1:
                self.has_house = True
            elif self.loan_purpose == 2:
                self.has_car = True
            elif self.loan_purpose == 3:
                self.education = min(4, self.education + 1)
                self.fin_lit = clamp(self.fin_lit + 0.05)
            elif self.loan_purpose == 4:
                self.has_job = True
                self.wealth += random.uniform(0, 10)
            self.credit_score = min(
                850, self.credit_score + random.uniform(5, 15) + 10 * self.fin_lit
            )  # improve credit
            self.loan_hist += 1
            self.trust = clamp(self.trust + 0.05)  # increased systemic / peer trust
        else:
            # negative effects (e.g., wealth loss, potential job loss, credit score dip)
            loss = random.uniform(0, 0.01 * self.loan_amount)
            self.wealth = max(1, self.wealth - loss)
            if not self.has_job and random.random() < 0.05 + 0.1 * (self.wealth < 20):
                pass  # if already jobless, no change
            elif self.has_job and random.random() < 0.02 + 0.05 * (self.wealth < 30):
                self.has_job = False
            self.credit_score = max(300, self.credit_score - random.uniform(0, 10))
            self.trust = clamp(self.trust - 0.05)  # decrease trust

        self.processed = True
        self.wealth = max(self.wealth, 1)
