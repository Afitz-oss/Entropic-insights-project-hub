import random
from faker import Faker
import pandas as pd
import textdistance
# Initialize a Faker instance to generate random names in English (US), Arabic, and Spanish (Spain) locales.
fake = Faker(['en_US', 'ar', 'es_ES'])

# A function to determine whether two names are similar based on the Jaro-Winkler and Levenshtein distance similarity thresholds.
def is_similar(name1, name2, threshold_jaro_winkler=0.85, threshold_levenshtein=0.75):
    jaro_winkler_similarity = textdistance.jaro_winkler(name1, name2)
    levenshtein_similarity = 1 - textdistance.levenshtein.normalized_distance(name1, name2)
    
    return jaro_winkler_similarity >= threshold_jaro_winkler or levenshtein_similarity >= threshold_levenshtein

# A function that introduces errors to a name by replacing, inserting, deleting, transposing characters, or deleting the middle name.
def introduce_error(name, n_modifications=1, p_replace=0.35, p_insert=0.25, p_delete=0.25, p_transpose=0.1, p_delete_middle_name=0.05):
    for _ in range(n_modifications):
        error_type = random.choices(["replace", "insert", "delete", "transpose", "delete_middle_name"],
                                     [p_replace, p_insert, p_delete, p_transpose, p_delete_middle_name])[0]
        if error_type == "delete_middle_name":
            name_parts = name.split()
            if len(name_parts) > 2:
                name = " ".join([name_parts[0], name_parts[-1]])
        else:
            index = random.randint(0, len(name) - 1)

            if error_type == "replace":
                new_char = random.choice(list(set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ ') - {name[index]}))
                name = name[:index] + new_char + name[index + 1:]
            elif error_type == "insert":
                new_char = random.choice('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ ')
                name = name[:index] + new_char + name[index:]
            elif error_type == "delete":
                name = name[:index] + name[index + 1:]
            else:  # error_type == "transpose"
                if index < len(name) - 1:
                    name = name[:index] + name[index + 1] + name[index] + name[index + 2:]

    return name

# A function to generate name pairs with matching and non-matching names. It uses the previous functions to create names with errors and unique names.
def generate_names(n_matches, repeats, n_unique, p_middle_name=0.3):
    matches = []
    for _ in range(n_matches):
        base_name = fake.name()
        if random.random() < p_middle_name:
            base_name = fake.name() + " " + fake.first_name()
        for _ in range(repeats):
            name_with_error = introduce_error(base_name, n_modifications=1)
            matches.append((base_name, name_with_error))

    uniques = [(fake.name(), fake.name()) for _ in range(n_unique)]

    dataset = matches + uniques
    random.shuffle(dataset)

    return dataset

# Sets the number of matches and the number of none matches and the number of repeats
n_matches = 3000
repeats = 2
n_unique = 10000

dataset = generate_names(n_matches, repeats, n_unique, p_middle_name=0.3)

# Creating a column called name number 1 and name number 2 
df = pd.DataFrame(dataset, columns=['Name Number 1','Name Number 2'])
# Creating a match column and hard coding it so that we can use a clssification model 
df['Match'] = df.apply(lambda row: 1 if is_similar(row['Name Number 1'], row['Name Number 2']) else 0, axis=1)

df.to_excel('name_matching_dataset.xlsx', index=False, engine='openpyxl')
