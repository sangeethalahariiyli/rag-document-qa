import sys
import os
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.utils.qa_chain import ask_question


def main():
    print("=" * 50)
    print("RAG Q&A Pipeline Test")
    print("=" * 50)

    questions = [
        "What is machine learning?",
        "What is the difference between supervised and unsupervised learning?",
        "What are neural networks?",
        "How do you prevent overfitting?",
    ]

    for i, question in enumerate(questions):
        print(f"\nQ: {question}")
        print("-" * 40)
        result = ask_question(question)
        print(f"A: {result['answer']}")
        print(f"\nSources used: {len(result['sources'])} chunks")

        # Wait between questions to avoid rate limit
        if i < len(questions) - 1:
            print("\nWaiting 15 seconds before next question...")
            time.sleep(30)

    print("\n" + "=" * 50)
    print("✅ All questions answered successfully!")


if __name__ == "__main__":
    main()