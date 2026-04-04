import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


class JsonTableHandler:
	#Base class for line-delimited JSON table files.
	def __init__(self, file_path: str):
		self.file_path = Path(file_path)
		self._table_definition: Optional[Dict[str, Any]] = None

    #Read lines from the table file.
	def read_lines(self):
		if not self.file_path.exists():
			return []

		with self.file_path.open("r", encoding="utf-8") as file:
			return [line.strip() for line in file if line.strip()]

    #load rows from the table file
	def load_rows(self):
		rows = []
		self._table_definition = None

		for line in self.read_lines():
			try:
				obj = json.loads(line)
			except json.JSONDecodeError:
				continue

			if isinstance(obj, dict) and "__TABLE_DEFINITION" in obj:
				self._table_definition = obj
			elif isinstance(obj, dict):
				rows.append(obj)

		return rows

    #Append and save rows to the table file.
	def save_rows(self, rows:List):
		output_lines = []

		if self._table_definition is not None:
			output_lines.append(json.dumps(self._table_definition, ensure_ascii=True))

		for row in rows:
			output_lines.append(json.dumps(row, ensure_ascii=True))

		self.file_path.parent.mkdir(parents=True, exist_ok=True)
		with self.file_path.open("w", encoding="utf-8") as file:
			file.write("\n".join(output_lines))
			if output_lines:
				file.write("\n")

	@staticmethod
	def next_id(rows:List):
		max_id = 0
		for row in rows:
			row_id = row.get("id")
			if isinstance(row_id, int) and row_id > max_id:
				max_id = row_id
		return max_id + 1


class AccountTableHandler(JsonTableHandler):
	def verify_username_password(self, username: str, password: str):
		#eturn user_id when username/password is valid, otherwise return None.
		rows = self.load_rows()
		for row in rows:
			if row.get("username") == username and row.get("password") == password:
				user_id = row.get("user_id")
				if isinstance(user_id, int):
					return user_id
		return None


class LoginRecordTableHandler(JsonTableHandler):
	def insert_login_record(self, user_id:int):
		#Insert a login record with current system time and return the inserted record.
		rows = self.load_rows()
		now = datetime.now().isoformat(timespec="seconds")

		record = {
            "computed": {},
			"stale": [],
			"id": self.next_id(rows),
			"createdAt": now,
			"updatedAt": now,
			"user_id": user_id
		}

		rows.append(record)
		self.save_rows(rows)
		return record

	def query_login_records_by_user_id(self, user_id:int):
		#Return login records for a user, each containing user_id and login_time.
		rows = self.load_rows()
		result = []

		for row in rows:
			if row.get("user_id") == user_id:
				login_time = row.get("createdAt")
				result.append({"user_id": user_id, "login_time": login_time})

		return result


@dataclass
class Quiz:
	quiz_id:int
	question: str
	correct_answer: str
	option_a: str
	option_b: str
	option_c: str
	option_d: str


class QuizBankPythonTableHandler(JsonTableHandler):
	def get_all_quiz(self) -> List[Quiz]:
		#Return all quiz rows as Quiz objects.
		rows = self.load_rows()
		quizzes = []

		for row in rows:
			quizzes.append(
				Quiz(
					quiz_id=int(row.get("id", 0)),
					question=str(row.get("question", "")),
					correct_answer=str(row.get("correct_answer", "")),
					option_a=str(row.get("option_a", "")),
					option_b=str(row.get("option_b", "")),
					option_c=str(row.get("option_c", "")),
					option_d=str(row.get("option_d", "")),
				)
			)

		return quizzes


class QuizRecordTableHandler(JsonTableHandler):
	def save_quiz_result(self, user_id:int, score):
		#Save a quiz result and return the inserted record.
		rows = self.load_rows()
		now = datetime.now().isoformat(timespec="seconds")

		record = {
			"computed": {},
			"stale": [],
			"id": self.next_id(rows),
			"createdAt": now,
			"updatedAt": now,
			"user_id": user_id,
			"score": str(score)
		}

		rows.append(record)
		self.save_rows(rows)
		return record

	def get_quiz_results_by_user_id(self, user_id:int):
		#Return all quiz results for a user with quiz_time.
		rows = self.load_rows()
		result = []

		for row in rows:
			if row.get("user_id") == user_id:
				quiz_time = row.get("createdAt")
				result.append(
					{
						"user_id": user_id,
						"score": row.get("score"),
						"quiz_time": quiz_time,
					}
				)

		return result