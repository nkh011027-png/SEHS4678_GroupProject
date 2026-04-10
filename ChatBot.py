import getpass
import sys
import random
import time
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from datetime import datetime
from pathlib import Path
from Database import AccountTableHandler, QuizBankPythonTableHandler, QuizRecordTableHandler, LoginRecordTableHandler
from nltk.stem.lancaster import LancasterStemmer
from IntentModel import IntentModel
sys.stdout.flush()
stemmer = LancasterStemmer()

TRAIN_MODEL = False

class ChatBotApp:
    account_handler:AccountTableHandler
    login_record_handler:LoginRecordTableHandler
    quiz_bank_handler:QuizBankPythonTableHandler
    quiz_record_handler:QuizRecordTableHandler
    current_user_id:int
    menu_intent_model:IntentModel
    chat_intent_model:IntentModel
    working_dir:str

    def __init__(self):
        self.working_dir = Path(__file__).parent
        self.account_handler = AccountTableHandler(f"{self.working_dir}/rsc/tables/AccountTable.json")
        self.login_record_handler = LoginRecordTableHandler(f"{self.working_dir}/rsc/tables/LoginRecordTable.json")
        self.quiz_bank_handler = QuizBankPythonTableHandler(f"{self.working_dir}/rsc/tables/QuizBankPythonTable.json")
        self.quiz_record_handler = QuizRecordTableHandler(f"{self.working_dir}/rsc/tables/QuizRecordTable.json")
        self.current_user_id = None
        # Initialize intent models for menu and chat functionalities, with options to force retraining and specify epochs and batch sizes for training.
        self.menu_intent_model = IntentModel(f"{self.working_dir}/rsc/intents/MenuIntents.json", force_retrain=TRAIN_MODEL, epochs_list=[500], batch_size_list=[16])
        self.chat_intent_model = IntentModel(f"{self.working_dir}/rsc/intents/ChatIntents.json", force_retrain=TRAIN_MODEL, epochs_list=[500], batch_size_list=[16])
        self.quiz_encourage_message = [
            "Keep up the good work!!!",
            "You're doing great, keep it up!!!",
            "Don't give up, you're almost there!!!",
            "Believe in yourself, you can do it!!!",
            "Every step you take is progress, keep going!!!"
        ]

    def run(self):
        while True:

            print("\nPolyU SPEED SEHS4678 NKH, CLS, WFW, WST")

            print("\n=== Login ===")
            username = input("Enter username: ")
            password = getpass.getpass("Enter password: ")

            user_id = self.account_handler.verify_username_password(username, password)
            if user_id is None:
                print("Username or password is incorrect.")
                continue

            self.current_user_id = user_id
            login_count = len(self.login_record_handler.query_login_records_by_user_id(self.current_user_id))
            if login_count > 0:
                self.slow_print(f"\nHappy to see you again, {username}!!", split_by_line=False, print_delay=0.02)
            else:
                self.slow_print(f"\nThis is the first time to see you, {username}!!", split_by_line=False, print_delay=0.02)

            
            self.login_record_handler.insert_login_record(user_id)
            self.show_main_menu()
            break

    def show_main_menu(self):
        while True:
            self.slow_print("\n=== Please choose ===\n1. Quiz me\n2. Encourage me\n3. Chat with me")
            inp = input("Enter your Choose: ")
            inp = inp.rstrip()
            choose = None
            responses = ""

            id_choose_map = {
                "1": "quiz",
                "2": "encourage",
                "3": "chat"
            }

            if inp in id_choose_map:
                choose = id_choose_map[inp]
                responses = random.choice(self.menu_intent_model.data["intents"][int(inp)-1]["responses"])
            else:
                predict_result = self.menu_intent_model.predict_intent(inp, result_tag=['responses', 'tag'])
                responses = predict_result[0] #responses
                choose = predict_result[1] #tag

            if choose == "quiz":
                self.start_quiz()
            elif choose == "encourage":
                self.start_encourage(responses)
            elif choose == "chat":
                self.start_chat()
    
    def start_chat(self):
        print("\n=== Chat ===")
        self.slow_print("Bot: Hello! I'm a chatbot, how can I help you today?", print_delay=0.02, split_by_line=False)
        while True:
            user_input = input("You: ").strip()
            predict_result = self.chat_intent_model.predict_intent(user_input, result_tag=['responses', 'tag'])

            if predict_result[1] == "hour":
                predict_result[0] = predict_result[0].replace("$time", datetime.now().strftime("%I:%M %p"))

            self.slow_print(f"Bot: {predict_result[0]}", split_by_line=False)

            if predict_result[1] == "goodbye":
                break

    def start_encourage(self, responses):
        extract_message = ""
        
        # This part of the code is used to extract the messsage by user login fequency
        login_records = self.login_record_handler.query_login_records_by_user_id(self.current_user_id)

        if len(login_records) > 0:
            if len(login_records) > 1:
                last_login_time_1 = datetime.strptime(login_records[-2]["login_time"], "%Y-%m-%dT%H:%M:%S")
                last_login_time_2 = datetime.strptime(login_records[-1]["login_time"], "%Y-%m-%dT%H:%M:%S")

                if (last_login_time_2 - last_login_time_1).total_seconds() < 60 * 24:
                    extract_message += f"I can see that you have logged in frequently with a short interval! That's great!! "

            extract_message += "Your last login time is {}. ".format(login_records[-1]["login_time"].replace("T", " "))

        # This part of the code is used to extract the message by user quiz performance
        quiz_records = self.quiz_record_handler.get_quiz_results_by_user_id(self.current_user_id)

        if len(quiz_records) > 0:
            soure_message = ""
            if len(quiz_records) > 1:
                last_quiz_score_1 = quiz_records[-2]["score"]
                last_quiz_score_2 = quiz_records[-1]["score"]

                if last_quiz_score_2 > last_quiz_score_1:
                    soure_message += f"I can see that your quiz score has improved from {last_quiz_score_1} to {last_quiz_score_2}! Keep it up!! "

            if soure_message == "":
                soure_message += f"Your latest quiz score is {quiz_records[-1]['score']}. "

            extract_message += soure_message

        if extract_message != "":
            responses = f"{extract_message}{responses}"

        self.slow_print(f"\nBot: {responses}", split_by_line=False, print_delay=0.002)
    
    #This method provide a quiz question include multiple-choice and fill-in-the-blank to user, and calculate the quiz score after user answer all questions. The quiz questions are randomly selected from the quiz bank, and the score is calculated based on the number of correct answers. And provides encouragement messages based on the user's quiz performance and login frequency, which are extracted from the user's quiz and login records. The quiz results are also saved to the database for future reference and analysis.
    def start_quiz(self):
        print("\n=== Quiz ===")
        quiz_bank = self.quiz_bank_handler.get_all_quiz()

        if not quiz_bank:
            print("No quiz questions available.")
            return
        
        total_questions = min(10, len(quiz_bank))
        selected_quizzes = random.sample(quiz_bank, total_questions)
        score = 0

        for index, quiz in enumerate(selected_quizzes, start=1):
            self.slow_print(f"\nQuestion {index}/{total_questions}:\n{quiz.question}")

            if quiz.option_a.strip() != "":
                self.slow_print(f"A. {quiz.option_a}\nB. {quiz.option_b}\nC. {quiz.option_c}\nD. {quiz.option_d}")

                while True:
                    user_input = input("Enter your answer (A/B/C/D): ").strip().upper()
                    if user_input in ["A", "B", "C", "D"]:
                        break
                    print("Please enter A, B, C, or D.")

                option_map = {
                    "A": quiz.option_a,
                    "B": quiz.option_b,
                    "C": quiz.option_c,
                    "D": quiz.option_d,
                }

                selected_answer = option_map[user_input].strip().lower()
                correct_answer = quiz.correct_answer.strip().lower()

                if selected_answer == correct_answer:
                    print("Correct!")
                    score += 1
                else:
                    print(f"Incorrect. Correct answer: {quiz.correct_answer}")
            else:
                user_input = input("Enter your answer: ").strip().lower()
                if user_input == quiz.correct_answer.strip().lower():
                    print("Correct!")
                    score += 1
                else:
                    print(f"Incorrect. Correct answer: {quiz.correct_answer}")

            self.slow_print(f"\nBot: {random.choice(self.quiz_encourage_message)}\n", split_by_line=False, print_delay=0.02)

        if score > 0:
            score = round(score / total_questions * 100)

        print(f"\nQuiz finished. Score: {score}")
        self.quiz_record_handler.save_quiz_result(self.current_user_id, score)
        
    def slow_print(self, text, split_by_line=True, print_delay=None):
        if split_by_line:
            delay=0.1
            if print_delay is not None:
                delay = print_delay

            split_text = text.split("\n")
            for line in split_text:
                print(line, end='\n', flush=True)
                time.sleep(delay)
        else:
            delay=0.01
            if print_delay is not None:
                delay = print_delay

            for char in text:
                print(char, end='', flush=True)
                time.sleep(delay)
        print()  # Final newline


if __name__ == "__main__":
    ChatBotApp().run()