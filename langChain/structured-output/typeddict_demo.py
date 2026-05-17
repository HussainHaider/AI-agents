from typing import TypedDict

class User(TypedDict):
    name: str
    age: int
    email: str

new_user: User = {
    "name": "Alice",
    "age": 30,
    "email": "test@test.com"
}

print(new_user)