from faker import Faker
import random
import csv
from datetime import datetime, timedelta

rowscount = 1_000_000
file_output = "medical_appointment.csv"

schedule_start = datetime(2010, 1, 1)
schedule_end = datetime(2025, 12, 31)

fake = Faker()

with open(file_output, mode="w", newline="", encoding="utf-8") as f:
    write = csv.writer(f)

    #heading
    write.writerow([
        "PatientId", "AppointmentID", "Gender",
        "ScheduledDay", "AppointmentDay", "Age",
        "Neighbourhood", "Scholarship", "Hipertension",
        "Diabetes", "Alcoholism", "Handcap",
        "SMS_received", "No-show"
    ])

    for i in range(rowscount):
        patient_id = random.randint(10**5, 10**7)
        appointment_id = random.randint(100_000, 10_000_000)

        gender = random.choice(["M", "F"])

        scheduled_day = fake.date_time_between(
            start_date = schedule_start,
            end_date = schedule_end
        )

        #appointment after scheduling
        appointment_day = scheduled_day + timedelta(days=random.randint(1, 30))

        age = random.randint(0, 90)

        #neighborhoodname
        neighbourhood = fake.city().upper()

        scholarship = random.choices([0, 1], weights=[80, 20])[0]
        hipertension = random.choices([0, 1], weights=[70, 30])[0]
        diabetes = random.choices([0, 1], weights=[85, 15])[0]
        alcoholism = random.choices([0, 1], weights=[95, 5])[0]
        handcap = random.choices([0, 1], weights=[98, 2])[0]
        sms_received = random.choices([0, 1], weights=[65, 35])[0]

        # No-show logic (more realistic)
        no_show = "Yes" if (
            sms_received == 0 and random.random() < 0.4
        ) else "No"

        write.writerow([
            patient_id,
            appointment_id,
            gender,
            scheduled_day.strftime("%d/%m/%Y %H:%M"),
            appointment_day.strftime("%d/%m/%Y %H:%M"),
            age,
            neighbourhood,
            scholarship,
            hipertension,
            diabetes,
            alcoholism,
            handcap,
            sms_received,
            no_show
        ])

print("✅ 1,000,000 rows generated (2010–2025, random neighbourhoods)")

