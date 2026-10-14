test = {
  "Population_and_Geography": ["population rank", "population distribution", "land area percentage", "main regions",
    "ethnicity", "official languages", "widely spoken languages", "famous river", "climate"],
  "Dates of Significance": ["national holidays", "religious holidays", "cultural holidays", "origin of festivals",
    "celebration of festivals", "symbol of festivals"],

    "Daily life and travel": ["payment habits", "travel habits", "bathing habits", "pet-raising habits", "social media usage pattern", "marriage customs"],
	"Diet and Health Concept": ["eating habits", "attitude towards birth, aging, illness and death", "traditional and modern medicine"],
	"Education and Knowledge": ["educational practices", "teacher student concept", "major selection", "career choices"],
	"Art and Entertainment": ["festive entertainment activities in different places with the same origin", "traditional music and dance", "contemporary art"],
	"Etiquette and Courtesy": [
		"basic etiquette", 
		"naming convention", "name origin", "name significance", "punctuality when visiting", "shoe etiquette during a visit", "hospitality customs when receiving guests", "bringing gifts for the host","seating etiquette for guests and hosts", "serving etiquette during a visit", 
		"leaving food after eating", "using salt while eating",
		"giving compliments during a meal", "eating with the right hand", "alcohol", "pork",
		"handing gifts", "gifts for children", "opening gifts",
		"general greeting principles", "greeting between men and men", "greeting between women and women", "greeting between men to women",
		"appointment scheduling before the business meeting", "dress code before the business meeting", "business card exchange before the business meeting", "network building before the business meeting", "age and experience before the business meeting", "familiarity before the business meeting",
		"socialization during the business meeting", "meeting duration during the business meeting", "open door policy during the business meeting", "interruptions during the business meeting", "deference to senior during the business meeting", "negotiation style during the business meeting", "decision making during the business meeting", "bartering during the business meeting", "private meetings during the business meeting", "confrontation avoidance during the business meeting",
		"follow up after the business meeting", "ongoing negotiations after the business meeting",
		"communication style", "indirect communication", "humour","physical contact", "personal space", "gestures", "beckoning", "eye contact"],
	"Fixed Expressions in Language": ["idiom", "common saying", "proverb", "neologism and abbreviation"],

    "Transportation Rules": ["Vehicle Movement Rules", "Traffic Signs and Signals", "Pedestrian and Non-Motorized Vehicle Rules", "Motor Vehicle Driving Rules", "Parking Regulations"],
    "Date Format": ["Date Order", "Separator Type", "Year Format", "Month Representation", "Zero-padding", "Natural Language Date", "Calendar System"],
    "Measurement Unit": ["Measurement System", "Localization"],
    "Financial Market Rules": ["Regulatory Structure", "Market Type", "Entity Type", "Access & Licensing", "Conduct & Compliance", "Capital & Risk", "Monetary & Payment System", "FinTech Regulation", "Tax & Accounting", "International Alignment", "Stability & Resolution"],

    "family_dynamics": ["communal living", "parental care"],
    "household_structures": ["authority", "patriarchy", "women roles in the family", "social interaction", "urban rural gap",
        "concept of seniority and childhood"],
    "gender roles": ["male dominance", "social compliance", "honour gender", "changing attitudes of gender"],

    "culture dimensions": ["power distance", "collectivism", "individualism", "motivation towards achievement and success", "uncertainty avoidance", "long term orientation", "indulgence"],
    "religion": ["religion"],
    "Do's and Don'ts": ["modest dress", "informality","compliments", "cultural acknowledgement", "education",
    "insults", "dirty jokes", "political criticism", "sensitive topics", "ethnicity assumptions", "stereotyping"]

}

print("the number of categories are: ",len(test)) 


import pandas as pd

# Load CSV
csv_path = r"C:\Users\Andres\Repos\Cultural_thesis\cultural_parameters\taxonomy_dimensions.csv"

df = pd.read_csv(csv_path)

# Adjust column name according to your CSV
csv_dimensions = set(df["Fine-grained Dimension"].tolist())

# Extract dimensions from your dictionary
test_dimensions = {
    dimension
    for category, dimensions in test.items()
    for dimension in dimensions
}


def normalize(text):
    return text.lower().strip()


csv_dimensions = {
    normalize(x)
    for x in df["Fine-grained Dimension"].tolist()
}

test_dimensions = {
    normalize(dimension)
    for category, dimensions in test.items()
    for dimension in dimensions
}

missing_in_csv = test_dimensions - csv_dimensions
missing_in_test = csv_dimensions - test_dimensions

print("Dimensions in dictionary but not in CSV:")
for dim in missing_in_csv:
    print(dim)

print("\nDimensions in CSV but not in dictionary:")
for dim in missing_in_test:
    print(dim)

print("\nCounts:")
print("CSV:", len(csv_dimensions))
print("Dictionary:", len(test_dimensions))


print("question.json counter:")
import json

json_path = r"C:\Users\Andres\Repos\Cultural_thesis\results\questions\questions.json"

with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

for culture, dimensions in data.items():
    print(f"{culture}: {len(dimensions)} dimensions")