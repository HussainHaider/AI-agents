from pydantic import BaseModel, EmailStr, Field
from typing import Optional

class Student(BaseModel):
    name: str = Field(..., description="The full name of the student")
    age: int = Field(..., gt=0, description="The age of the student, must be a positive integer")
    email: EmailStr = Field(..., description="The email address of the student")
    grade: Optional[str] = Field(None, description="The grade of the student, e.g., A, B, C")
    cgpa: float = Field(gt=0, lt=10, default=5, description='A decimal value representing the cgpa of the student')


new_student = {'age': 32, 'email':'abc@gmail.com', 'name':'Hussain Haider', 'grade':'A+', 'cgpa': 9.5}

student = Student(**new_student)
print(student)

student_dict = dict(student)

print(student_dict['age'])

student_json = student.model_dump_json()