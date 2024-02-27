# coding: utf-8
from mrjob.job import MRJob
from mrjob.step import MRStep

class MRTitanicAnalysis(MRJob):

    def steps(self):
        return [
            MRStep(mapper=self.mapper_get_gender_survived,
                   reducer=self.reducer_calculate_survival_rate_by_gender),
            MRStep(mapper=self.mapper_get_class_fare,
                    reducer=self.reducer_calculate_average_fare_by_class)
        ]

    def parse_line(self, line):
        # Custom parsing logic to handle commas within quotes for names
        fields = []
        field_start = 0
        in_quotes = False
        for i, char in enumerate(line):
            if char == '"' and line[i-1] != '\\':  # Ignore escaped quotes
                in_quotes = not in_quotes
            elif char == ',' and not in_quotes:
                fields.append(line[field_start:i])
                field_start = i + 1
        fields.append(line[field_start:])  # Add the last field
        return fields
    
    # Step 1: Survival Rate by Gender
    def mapper_get_gender_survived(self, _, line):
        columns = self.parse_line(line)
        if columns[0] == "PassengerId":  # Skip header
            return
        gender = columns[4]  # Assuming gender is in the 5th column
        survived = columns[1]  # Assuming survival status is in the 2nd column
        yield ('Gender-Survived', (gender, int(survived)))

    def reducer_calculate_survival_rate_by_gender(self, key, values):
        survival_counts = {'male': [0, 0], 'female': [0, 0]}  # [survived, total]
        for gender, survived in values:
            survival_counts[gender][0] += survived
            survival_counts[gender][1] += 1
        for gender in survival_counts:
            yield (gender, survival_counts[gender][0] / survival_counts[gender][1])
    
    # Step 2: Average Fare by Class
    def mapper_get_class_fare(self, _, line):
        columns = self.parse_line(line)
        if columns[0] == "PassengerId":  # Skip header
            return
        pclass = columns[2]  # Assuming passenger class is in the 3rd column
        fare = columns[9]  # Assuming fare is in the 10th column
        try:
            yield ('Class-Fare', (pclass, float(fare)))
        except ValueError:  # Skip rows with invalid fare data
            return

    def reducer_calculate_average_fare_by_class(self, key, values):
        fare_sum = {'1': [0, 0], '2': [0, 0], '3': [0, 0]}  # [total fare, count]
        for pclass, fare in values:
            fare_sum[pclass][0] += fare
            fare_sum[pclass][1] += 1
        for pclass in fare_sum:
            yield (pclass, fare_sum[pclass][0] / fare_sum[pclass][1])

if __name__ == '__main__':
    MRTitanicAnalysis.run()
