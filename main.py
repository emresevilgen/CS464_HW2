import math

selection = -math.inf

while selection != -1:
    selection = input(
        "\n1) Question 1 \n2) Question 2 \n3) Question 3 \n4) Question 4 \n6) Quit \n\nEnter your selection (1, 2, ..., 6): ")
    try:
        selection = int(selection)
    except:
        selection = -math.inf
    if selection == 1:
        print("\nQuestion 1 is running.\n------------------------")
        exec(open("question_1.py").read())
        print("Question 1 is done.\n")
    elif selection == 2:
        print("\nQuestion 2 is running.\n------------------------")
        exec(open("question_2.py").read())
        print("Question 2 is done.\n")
    elif selection == 3:
        print("\nQuestion 3 is running.\n------------------------")
        exec(open("question_3.py").read())
        print("Question 3 is done.\n")
    elif selection == 4:
        print("\nQuestion 4 is running.\n------------------------")
        exec(open("question_4.py").read())
        print("Question 4 is done.\n")
    elif selection == 6:
        selection = -1
    else:
        print("\nInvalid selection.")
        selection = -math.inf
