import os
import random
from datetime import datetime
from faker import Faker 
import sys

fake = Faker()

def random_dob(min_year=1960, max_year=2005):
    year = random.randint(min_year, max_year)
    month = random.randint(1, 12)
    day = random.randint(1, 28)
    return f"{day:02d}/{month:02d}/{year:04d}"

def random_gender():
    return random.choice(["Male", "Female", "Other", "Prefer not to say"])

def random_phone_number():
    area_code = random.randint(100,999)
    rest_of_phone = area_code + random.randint(1000000, 9999999)
    return rest_of_phone

def random_address():
    return fake.address()


def random_employer():
    return fake.company()
    

def random_associates():
    return fake.name()

def random_name():
    return fake.name()


def create_txt_files(file_name, name):
    with open(file_name, "w") as f:
        f.write("Name: " + name + "\n")
        f.write("Date of Birth: " + random_dob() + "\n")
        f.write("Gender: " + random_gender() + "\n")
        f.write('Possible Phones Associated with Subject:' + str(random_phone_number()) + "\n")
        f.write('Indicators' + "\n")
        f.write('Possible Employers' + random_employer() + "\n")
        f.write('Address Summary ' + random_address() + "\n")
        f.write('Address Details' + "\n")
        f.write('Possible Associates - Summary' + random_associates() + "\n")
        f.write('Possible Associates - Details' + "\n")


def main(num_files):
    text_file_type = "Comprehensive"
    report_type = "Report"
    timestamp = int(datetime.now().timestamp()* 1000)

    for i in range(num_files):
        name = random_name()
        timestamp = int(datetime.now().timestamp()* 1000)
        file_name = f"{(name)}-{text_file_type}-{report_type}-{timestamp}{i}.txt"
        create_txt_files(file_name, name)





if __name__ == "__main__":
    # Set the number of files you want to generate here
    num_files = 100

    main(num_files)