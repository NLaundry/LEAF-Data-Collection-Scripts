import json
import random

def generate_synthetic_data(N):
    data = []
    
    majors = ["Computer Science", "Social Science", "Liberal Arts", "non-Computer Science STEM degree"]
    
    for i in range(N):
        major = random.choice(majors)
        
        # Biases based on major
        if major == "Computer Science":
            cs_perception = random.randint(4, 5)
            concept_relevance = random.randint(3, 5)
            personalized_boost = 1  # slight boost for personalized content
        else:  
            cs_perception = random.randint(2, 4)
            concept_relevance = random.randint(2, 4)
            personalized_boost = 2  # more pronounced boost for personalized content for non-CS majors
        
        expertise = random.randint(0, 10)  # 0 to 10 years of expertise
        
        # Biases based on expertise
        better_understanding = min(5, expertise // 2 + 2)
        teach_concept = min(5, expertise // 2 + 1)
        
        # Creating a function to generate content values
        def generate_content_values(base_value, boost):
            return base_value, min(5, base_value + boost)
        
        entry = {
            "participant_id": f"P{str(i+1).zfill(3)}",
            "demographics": {
                "student_major": major,
                "prior_expertise": expertise,
                "perception_of_CS_value": cs_perception
            },
            "generic_content": {
                "P3": {
                    "personalize_effectively": random.randint(1, 4),
                    "text_specific_content": random.randint(1, 4)
                },
                "C1": {
                    "concept_relevance": concept_relevance,
                    "concept_value_learning": random.randint(1, 4)
                },
                "RQ4": {
                    "CS_relevance": cs_perception,
                    "CS_value_learning": cs_perception,
                    "CS_useful_career": random.randint(1, 4)
                },
                "C2": {
                    "better_understanding": better_understanding,
                    "teach_concept": teach_concept,
                    "succeed_applying_concept": random.randint(1, 4)
                },
                "C3": {
                    "effectiveness": random.randint(1, 4),
                    "clarity": random.randint(1, 4)
                }
            }
        }
        
        # Now add personalized content with scores boosted
        entry["personalized_content"] = {}
        for key, values in entry["generic_content"].items():
            entry["personalized_content"][key] = {}
            for sub_key, value in values.items():
                generic_value, personalized_value = generate_content_values(value, personalized_boost)
                entry["generic_content"][key][sub_key] = generic_value
                entry["personalized_content"][key][sub_key] = personalized_value

        data.append(entry)

    return data

# Set N (number of synthetic data points)
N = 1000
data = generate_synthetic_data(N)

# Save to a JSON file
with open("synthetic_data.json", "w") as f:
    json.dump(data, f, indent=4)

print(f"{N} synthetic data points saved to 'synthetic_data.json'")
