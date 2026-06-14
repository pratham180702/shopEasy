import requests
import pandas as pd
from datasets import Dataset

from ragas import evaluate
from ragas.metrics import answer_relevancy


API_URL = "http://localhost:8000/trigger_rag"


TEST_CASES = [
    {
        "question": "What countries does ShopEasy operate in?",
        "ground_truth": "ShopEasy operates in India, the United States, the United Kingdom, and Canada."
    },
    {
        "question": "How can I contact ShopEasy support?",
        "ground_truth": "Customers can contact ShopEasy via 24/7 Live Chat, email at support@shopeasy.com, phone from 9 AM to 9 PM local time, or the Help Center."
    },
    {
        "question": "Is it free to create a ShopEasy account?",
        "ground_truth": "Yes, creating a standard ShopEasy account is free. ShopEasy Plus is a paid subscription."
    },
    {
        "question": "When is KYC required on ShopEasy?",
        "ground_truth": "KYC is required for high-value orders, seller program enrollment, EMI or Buy Now Pay Later options, redeeming loyalty points above a threshold, or fraud-flagged accounts."
    },
    {
        "question": "How long does KYC verification take?",
        "ground_truth": "KYC verification usually takes 1 to 3 business days."
    },
    {
        "question": "What happens after 5 failed login attempts?",
        "ground_truth": "After 5 consecutive failed login attempts, the account is temporarily locked for 30 minutes."
    },
    {
        "question": "How long is the password reset link valid?",
        "ground_truth": "The password reset link is valid for 60 minutes."
    },
    {
        "question": "How many delivery addresses can I save on ShopEasy?",
        "ground_truth": "Customers can save up to 10 delivery addresses and 3 billing addresses."
    },
    {
        "question": "Can I change my delivery address after placing an order?",
        "ground_truth": "Address changes are possible only within 30 minutes of placing the order and only if the order has not been processed for shipping."
    },
    {
        "question": "How long do items remain in the shopping cart?",
        "ground_truth": "Items remain in the cart for up to 30 days for logged-in users, or until the browser session ends for guest users."
    },
    {
        "question": "Which payment methods are supported by ShopEasy?",
        "ground_truth": "ShopEasy supports credit/debit cards, net banking, UPI, PayPal, Apple Pay, Google Pay, Cash on Delivery in select Indian PIN codes, ShopEasy Wallet, Gift Cards, and EMI or Buy Now Pay Later options."
    },
    {
        "question": "Can I use multiple coupon codes on one order?",
        "ground_truth": "Generally, only one coupon code can be applied per order, but Gift Cards and Store Credits can be combined with a coupon."
    },
    {
        "question": "How long are ShopEasy Gift Cards valid?",
        "ground_truth": "ShopEasy Gift Cards are valid for 12 months from the date of purchase."
    },
    {
        "question": "How can I track my ShopEasy order?",
        "ground_truth": "Customers can track orders from Account > Orders by selecting the Order ID and clicking Track Shipment. They can also use the tracking link in the shipment notification email."
    },
    {
        "question": "What should I do if my order has no tracking movement for 48 hours?",
        "ground_truth": "If tracking shows no movement for 48 hours, the customer should contact support. ShopEasy will initiate a courier trace within 1 business day."
    },
    {
        "question": "How many delivery attempts does ShopEasy make?",
        "ground_truth": "ShopEasy makes up to 3 delivery attempts on consecutive business days."
    },
    {
        "question": "What is ShopEasy's standard return window?",
        "ground_truth": "ShopEasy's standard return window is 10 days from the date of delivery for most product categories."
    },
    {
        "question": "What is the return window for fashion and footwear items?",
        "ground_truth": "Fashion and footwear items have a 30-day return window, provided they are unworn and meet the return conditions."
    },
    {
        "question": "Are digital products returnable on ShopEasy?",
        "ground_truth": "Digital products are non-returnable once delivered or activated. If defective, customers should contact support within 7 days for a replacement or refund."
    },
    {
        "question": "How long does a refund take?",
        "ground_truth": "Card or bank refunds take 5 to 7 business days after approval. UPI or net banking refunds take 3 to 5 business days. ShopEasy Wallet refunds are processed within 24 hours."
    },
    {
        "question": "When should a customer escalate a support issue?",
        "ground_truth": "Escalation is recommended when an issue remains unresolved after 3 contacts, an approved refund is not received after 10 business days, seller disputes remain unresolved after 7 days, or the matter involves fraud, suspension, safety, or legal issues."
    },
    {
        "question": "How do I escalate a ShopEasy support issue?",
        "ground_truth": "Customers can escalate by contacting Live Chat or emailing support@shopeasy.com with ESCALATE in the subject line, along with account email, Order ID, Case Number, and issue summary."
    }
]


def get_api_response(question: str, session_id: str) -> str:
    payload = {
        "query": question,
        "session_id": session_id
    }

    response = requests.post(
        API_URL,
        json=payload,
        timeout=120
    )

    response.raise_for_status()

    data = response.json()
    return data.get("response", "")


def run_evaluation():
    rows = []

    for index, test_case in enumerate(TEST_CASES):
        question = test_case["question"]
        ground_truth = test_case["ground_truth"]

        answer = get_api_response(
            question=question,
            session_id=f"eval-session-{index}"
        )

        rows.append({
            "question": question,
            "answer": answer,
            "ground_truth": ground_truth,
        })

    dataset = Dataset.from_list(rows)

    result = evaluate(
        dataset,
        metrics=[
            answer_relevancy,
        ],
    )

    df = result.to_pandas()
    df.to_csv("evaluation_results.csv", index=False)

    print(df)

    print("\nAverage Scores:")
    print(df[["answer_relevancy"]].mean())

    return df


if __name__ == "__main__":
    run_evaluation()