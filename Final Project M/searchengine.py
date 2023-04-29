import argparse
from utilities import handlers


def main(args):
    if args.option == 1:
        handlers.collect_new_documents()
    elif args.option == 2:
        handlers.index_documents()
    elif args.option == 3:
        handlers.search_for_query(args.query)
    elif args.option == 4:
        handlers.train_ml_classifier()
    elif args.option == 5:
        handlers.predict_link(args.link)
    elif args.option == 6:
        handlers.your_story()
    else:
        print("Invalid choice. Please enter a number between 1 and 6.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Search Engine options")
    parser.add_argument("option", type=int, help="Choose an option between 1 and 6.")
    parser.add_argument("--query", type=str, help="Query for option 3.", default="")
    parser.add_argument("--link", type=str, help="Link for option 5.", default="")
    args = parser.parse_args()
    main(args)
