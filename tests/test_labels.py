"""R-13 snapshot builder on five hand-built toy loans; R-15 at-risk; R-16 censoring."""
import numpy as np
import pandas as pd

from creditpulse.labels.events import build_snapshots, month_index
from creditpulse.utils import load_config

CFG = load_config()
END = month_index(pd.Timestamp(CFG["data"]["data_end"]))  # 1998-12


def toy(loan_id, start, dur, event=None):
    return {"loan_id": loan_id, "account_id": loan_id, "start_mi": start, "first_due_mi": start + 1,
            "last_due_mi": start + dur, "event_mi": np.nan if event is None else start + event,
            "has_event": int(event is not None)}


def snaps(*loans):
    return build_snapshots(pd.DataFrame(loans), CFG)


def test_clean_loan_all_negative_until_last_instalment():
    s = snaps(toy(1, END - 40, 24))
    assert s.loan_age.min() == 3 and s.loan_age.max() == 23
    assert (s.label == 0).all() and (s.event_k == 0).all()


def test_event_loan_labels_and_event_k():
    s = snaps(toy(2, END - 40, 36, event=8))
    assert s.loan_age.tolist() == [3, 4, 5, 6, 7]                 # stops before the event (D2)
    assert (s.label == 1).all()                                     # every cutoff is within 6 months
    assert s.event_k.tolist() == [5, 4, 3, 2, 1]


def test_event_before_first_cutoff_gives_no_snapshots():
    assert len(snaps(toy(3, END - 40, 24, event=2))) == 0


def test_right_censoring_at_data_end():
    s = snaps(toy(4, END - 10, 60))
    tail = s[s.cutoff_mi + 6 > END]
    assert tail.label.isna().all()                                  # R-16: never a negative
    assert (tail.follow_k == END - tail.cutoff_mi).all()
    assert (s[s.cutoff_mi + 6 <= END].label == 0).all()


def test_repaid_inside_window_is_negative_not_censored():
    s = snaps(toy(5, END - 20, 12))                                 # last instalment 8 months before data end
    assert s.label.notna().all() and (s.label == 0).all()
    last = s.iloc[-1]
    assert last.follow_k == 1                                       # observed until the final instalment


def test_event_late_in_window():
    s = snaps(toy(6, END - 40, 48, event=20))
    lab = s.set_index("loan_age").label
    assert lab.loc[13] == 0 and lab.loc[14] == 1 and lab.loc[19] == 1
