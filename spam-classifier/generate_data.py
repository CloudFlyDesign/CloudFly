"""Generate a synthetic spam / ham email dataset (1000 samples)."""

from __future__ import annotations

import csv
import random
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"
OUTPUT_PATH = DATA_DIR / "emails.csv"
N_SAMPLES = 1000
SEED = 42

SPAM_SUBJECTS = [
    "Congratulations! You've won a prize",
    "URGENT: Verify your account now",
    "Claim your free gift card today",
    "Limited time offer - act now!!!",
    "You have been selected for a cash reward",
    "Cheap meds online - no prescription",
    "Make $5000 from home this week",
    "Your package could not be delivered - click here",
    "Reset your password immediately",
    "Exclusive deal just for you",
    "Lottery winner notification",
    "Double your income overnight",
    "Hot singles in your area",
    "Refinance your loan at 0% APR",
    "Free trial - cancel anytime (not really)",
]

SPAM_BODIES = [
    "Click the link below to claim your reward before it expires. Act now!!!",
    "Dear customer, we detected unusual activity. Verify your identity at this link.",
    "You have been pre-approved. Reply with your bank details to proceed.",
    "Congratulations winner! Send a small processing fee to unlock your prize.",
    "Limited stock remaining. Buy now and get 90% off. No questions asked.",
    "Work from home and earn thousands. No experience needed. Sign up free.",
    "Your account will be suspended unless you confirm your password today.",
    "Miracle weight loss pills - lose 30 pounds in 7 days. Order now.",
    "Transfer inheritance funds internationally. Contact our agent ASAP.",
    "Nigerian prince seeking trusted partner for large fund transfer.",
    "Free crypto tokens waiting in your wallet. Connect now to claim.",
    "Your invoice is overdue. Pay immediately via this secure link.",
    "Exclusive membership unlocked. Enter credit card to activate free trial.",
    "Hot investment tip: guaranteed returns of 400%. Join our VIP group.",
    "We tried to deliver your parcel. Reschedule delivery by clicking here.",
]

HAM_SUBJECTS = [
    "Team standup notes from Monday",
    "Lunch plans for Friday?",
    "Project timeline update",
    "Re: Q3 budget draft",
    "Meeting invite: design review",
    "Thanks for your help yesterday",
    "Flight itinerary confirmation",
    "Homework assignment reminder",
    "Weekend hiking trip",
    "Invoice for consulting work",
    "Doctor appointment confirmation",
    "Book club reading list",
    "Office wifi password reset (internal)",
    "Birthday party this Saturday",
    "Code review request for PR #142",
]

HAM_BODIES = [
    "Hi, can we move our 1:1 to Thursday afternoon? Let me know what works.",
    "Attached are the notes from today's standup. Action items are listed below.",
    "Thanks for sending the draft. I left a few comments in the doc.",
    "Reminder: the design review is at 2pm in conference room B.",
    "Looking forward to catching up this weekend. Does 11am still work?",
    "Please find the updated project timeline attached. Feedback welcome.",
    "Your appointment is confirmed for Tuesday at 10:30am. Reply to reschedule.",
    "I pushed the latest changes to the feature branch. Ready for review.",
    "Mom said dinner is at 6. Can you pick up bread on the way?",
    "The client approved the proposal. Next step is the kickoff meeting.",
    "Here is the reading list for next month's book club. See you Thursday.",
    "Flight UA482 departs at 8:15am. Gate info will update closer to departure.",
    "Sharing the Q3 numbers we discussed. Happy to walk through them tomorrow.",
    "Could you send me the link to last week's recording when you get a chance?",
    "Great work on the presentation. The stakeholders were impressed.",
]

FILLERS = [
    "Please let me know if you have questions.",
    "Looking forward to your reply.",
    "Best regards.",
    "Thanks in advance.",
    "Have a great day.",
    "Talk soon.",
    "Cheers.",
]


def _compose(subjects: list[str], bodies: list[str], rng: random.Random) -> str:
    subject = rng.choice(subjects)
    body = rng.choice(bodies)
    if rng.random() < 0.6:
        body = f"{body} {rng.choice(FILLERS)}"
    if rng.random() < 0.3:
        body = f"{body} Ref #{rng.randint(1000, 9999)}."
    return f"Subject: {subject}\n\n{body}"


def generate_dataset(n: int = N_SAMPLES, seed: int = SEED) -> list[dict[str, str | int]]:
    rng = random.Random(seed)
    rows: list[dict[str, str | int]] = []
    # Roughly balanced classes
    n_spam = n // 2
    n_ham = n - n_spam

    for _ in range(n_spam):
        rows.append({"text": _compose(SPAM_SUBJECTS, SPAM_BODIES, rng), "label": 1})
    for _ in range(n_ham):
        rows.append({"text": _compose(HAM_SUBJECTS, HAM_BODIES, rng), "label": 0})

    rng.shuffle(rows)
    return rows


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    rows = generate_dataset()
    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "label"])
        writer.writeheader()
        writer.writerows(rows)
    spam = sum(1 for r in rows if r["label"] == 1)
    print(f"Wrote {len(rows)} emails to {OUTPUT_PATH}")
    print(f"  spam={spam}, ham={len(rows) - spam}")


if __name__ == "__main__":
    main()
