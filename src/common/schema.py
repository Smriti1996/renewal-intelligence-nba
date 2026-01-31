from __future__ import annotations
from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class MembersSchema:
    membership_nbr: str = "membership_nbr"
    persona_id: str = "persona_id"
    tenure_bucket: str = "tenure_bucket"
    membership_tier: str = "membership_tier"
    membership_type: str = "membership_type"
    auto_renew_opt_in: str = "auto_renew_opt_in"
    sales_decile: str = "sales_decile"
    sales_centile: str = "sales_centile"
    tenure_months: str = "tenure_months"

    def columns(self) -> List[str]:
        return [
            self.membership_nbr,
            self.persona_id,
            self.tenure_bucket,
            self.membership_tier,
            self.membership_type,
            self.auto_renew_opt_in,
            self.sales_decile,
            self.sales_centile,
            self.tenure_months,
        ]


@dataclass(frozen=True)
class NbaUpliftSchema:
    persona_id: str = "persona_id"
    tenure_bucket: str = "tenure_bucket"
    entity_type: str = "entity_type"
    entity_id: str = "entity_id"
    entity_name: str = "entity_name"
    n_test_matched: str = "n_test_matched"
    n_control_matched: str = "n_control_matched"
    test_renewal_rate: str = "test_renewal_rate"
    control_renewal_rate: str = "control_renewal_rate"
    incremental_renewal_rate: str = "incremental_renewal_rate"
    incremental_rank: str = "incremental_rank"
    uplift_method: str = "uplift_method"

    def columns(self) -> List[str]:
        return [
            self.persona_id,
            self.tenure_bucket,
            self.entity_type,
            self.entity_id,
            self.entity_name,
            self.n_test_matched,
            self.n_control_matched,
            self.test_renewal_rate,
            self.control_renewal_rate,
            self.incremental_renewal_rate,
            self.incremental_rank,
            self.uplift_method,
        ]
