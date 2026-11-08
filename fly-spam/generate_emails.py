"""Create a synthetic 1000-email spam/ham dataset."""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

SPAM_SUBJECTS = [
    "You won a $10,000 prize!!!",
    "URGENT: verify your account now",
    "Limited time offer — act immediately",
    "Congratulations, claim your free gift card",
    "Your package is on hold — confirm payment",
    "Make money working from home",
    "Hot singles in your area",
    "Final notice: account suspension",
    "Crypto investment opportunity 500% returns",
    "Click here to unlock exclusive deal",
]

HAM_SUBJECTS = [
    "Team standup notes for Monday",
    "Lunch tomorrow?",
    "Invoice #4821 attached",
    "Project timeline update",
    "Thanks for the review comments",
    "Flight itinerary for Thursday",
    "Weekly status report",
    "Can we reschedule the call?",
    "Docs for the Q3 budget",
    "Happy birthday!",
]

SPAM_BODIES = [
    "Click the link below to claim your prize before it expires. Wire fees may apply. Act now!!!",
    "We detected unusual login activity. Verify your password at this secure page immediately.",
    "Only 3 spots left. Double your income this week with our proven system. Reply YES.",
    "Your payment failed. Update your credit card details within 24 hours to avoid suspension.",
    "Exclusive crypto drop. Guaranteed returns. Send Bitcoin to the address in this email.",
    "Congratulations winner! You have been selected. Confirm shipping and SSN to proceed.",
    "Lose 30 pounds this month with this one weird trick. Limited stock — buy now.",
    "Your inheritance of $2,500,000 is ready. Contact our agent to release the funds.",
    "Free iPhone giveaway. Complete a short survey and pay a small shipping fee.",
    "Account locked. Restore access by confirming your bank login on our portal.",
]

HAM_BODIES = [
    "Hi, attaching the slides from today's meeting. Let me know if you need edits.",
    "Are you free for coffee around 2pm? We can go over the design feedback.",
    "Please review the draft proposal by Friday. I already incorporated Sam's notes.",
    "The deploy finished successfully. Monitoring looks green on staging.",
    "Reminder: dentist appointment next Tuesday at 10:30. Reply if you need to move it.",
    "Thanks for sending the contract. I signed and returned it this morning.",
    "Kids soccer practice ends at 5. Can you pick them up if I'm still in meetings?",
    "Here is the link to the shared folder with the photos from the retreat.",
    "Quick note: the printer on floor 3 is jammed again. IT already has a ticket.",
    "Looking forward to the conference. See you at the registration desk.",
]


def make_email(rng: random.Random, label: str) -> dict:
    if label == "spam":
        subject = rng.choice(SPAM_SUBJECTS)
        body = rng.choice(SPAM_BODIES)
        # light variation so samples aren't identical
        body = f"{body} Ref:{rng.randint(1000, 9999)}"
    else:
        subject = rng.choice(HAM_SUBJECTS)
        body = rng.choice(HAM_BODIES)
        body = f"{body} Ticket:{rng.randint(1000, 9999)}"
    return {
        "id": None,
        "subject": subject,
        "body": body,
        "text": f"{subject}\n\n{body}",
        "label": label,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "data" / "emails.csv",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)
    n_spam = args.n // 2
    n_ham = args.n - n_spam
    rows = [make_email(rng, "spam") for _ in range(n_spam)]
    rows += [make_email(rng, "ham") for _ in range(n_ham)]
    rng.shuffle(rows)
    for i, row in enumerate(rows):
        row["id"] = i

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "subject", "body", "text", "label"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} emails -> {args.out}")


if __name__ == "__main__":
    main()
