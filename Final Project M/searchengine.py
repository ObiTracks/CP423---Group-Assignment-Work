from utilities import handlers

def main_menu():
    print("Select an option:")
    print("1- Collect new documents.")
    print("2- Index documents.")
    print("3- Search for a query.")
    print("4- Train ML classifier.")
    print("5- Predict a link.")
    print("6- Your story!")
    print("7- Exit")

def main():
    while True:
        main_menu()
        user_choice = int(input("Enter your choice (1-7): "))

        if user_choice == 1:
            handlers.collect_new_documents()
        elif user_choice == 2:
            handlers.index_documents()
        elif user_choice == 3:
            handlers.search_for_query()
        elif user_choice == 4:
            handlers.train_ml_classifier()
        elif user_choice == 5:
            handlers.predict_link()
        elif user_choice == 6:
            handlers.your_story()
        elif user_choice == 7:
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please enter a number between 1 and 7.")

if __name__ == "__main__":
    main()
