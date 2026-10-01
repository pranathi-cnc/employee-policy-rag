from app import load_documents, answer_question


questions = [
    "How many casual leave days are employees entitled to?",
    "How many sick leave days are employees entitled to?",
    "Can unused casual leave be carried forward?",
    "How many days in advance should casual leave normally be requested?",
    "What is the maximum meal reimbursement per day?",
    "Within how many days must reimbursement claims be submitted?",
    "Are receipts required for reimbursement?",
    "What are the standard working hours?",
    "How long is the lunch break?",
    "Should remote employees remain available during standard working hours?",
    "How many work-from-home days are allowed per month?",
    "What is the annual bonus percentage for employees?",
]


def evaluate():

    load_documents()

    print("\n" + "=" * 80)
    print("EMPLOYEE POLICY QA EVALUATION")
    print("=" * 80)

    for number, question in enumerate(questions, start=1):

        answer, retrieved_chunks = answer_question(question)

        sources = sorted(
            set(item["source"] for item in retrieved_chunks)
        )

        print(f"\n{number}. Question:")
        print(question)

        print("\nAnswer:")
        print(answer)

        print("\nRetrieved Sources:")
        for source in sources:
            print(f"- {source}")

        print("-" * 80)


if __name__ == "__main__":
    evaluate()