from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load environment variables from .env file
load_dotenv()

# Correctly initialize the embedding model with the required prefix
embedding = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

# Correct the documents list by adding commas to separate each string
documents = [
    "Starts the day with a hot cup of chai and a quick check of emails on their phone.",
    "Navigates through heavy traffic, often relying on public transport or a shared cab.",
    "Attends the daily stand-up meeting, updating the team on project progress.",
    "Sits in an air-conditioned office, typing away at a computer.",
    "Takes a quick break to grab a samosa or vada pav from a nearby stall.",
    "Collaborates with colleagues on a project, often switching between Hindi and English.",
    "Has a packed lunch from home, sharing a bit with their colleagues.",
    "Attends a long, often draining, meeting with a manager or client.",
    "Tries to finish a pending task before the day ends, fighting the post-lunch slump.",
    "Engages in office gossip, discussing weekend plans or a new movie release.",
    "Deals with a technical issue, often calling the IT support team.",
    "Works late to meet a tight deadline, ordering a quick dinner from a food app.",
    "Commutes back home, listening to music or a podcast to unwind.",
    "Spends the evening with family, helping with chores or watching TV.",
    "Juggles work-life balance, often taking work calls after office hours.",
    "Plans a weekend trip to a nearby hill station or beach.",
    "Fills out a timesheet, accurately logging their hours.",
    "Participates in a team-building activity, like a cricket match or a game of carrom.",
    "Deals with a power cut, relying on the office generator.",
    "Takes a short nap during a long meeting, hoping no one notices.",
    "Complains about the office politics and the never-ending bureaucracy.",
    "Attends a company party, socializing with colleagues outside of work.",
    "Learns a new skill to stay relevant in a fast-paced work environment.",
    "Celebrates a colleague's birthday with a small cake and a song.",
    "Travels for work, visiting a client or another branch in a different city.",
    "Spends hours on conference calls, often with international clients.",
    "Gets stuck in a traffic jam, arriving late to the office.",
    "Takes a day off to attend a family function or a festival.",
    "Spends time on social media, scrolling through Instagram and LinkedIn.",
    "Prepares a presentation for a client, spending hours on a single slide.",
    "Deals with a difficult boss, learning to navigate their moods.",
    "Finds a new job, looking for better opportunities and a higher salary.",
    "Works on a weekend to finish a project, hoping for a day in lieu.",
    "Spends hours on the phone, trying to resolve a customer's issue.",
    "Eats a healthy salad for lunch, hoping to lose a few pounds.",
    "Takes a quick trip to the pantry, grabbing a cup of coffee and a cookie.",
    "Works from home, dealing with distractions from family and pets.",
    "Plans a team lunch, trying to find a restaurant that everyone likes.",
    "Attends a training session, learning about a new software or technology.",
    "Struggles to stay awake during a long, boring training session.",
    "Complains about the office temperature, which is either too hot or too cold.",
    "Deals with a difficult colleague, trying to maintain a professional relationship.",
    "Attends a farewell party for a colleague, wishing them well on their new journey.",
    "Spends hours on a single task, trying to get every detail right.",
    "Takes a day off to relax and recharge, often staying at home.",
    "Works on a side project, hoping to turn it into a full-time business.",
    "Spends hours on a single email, trying to get the wording just right.",
    "Deals with a slow internet connection, wasting valuable time.",
    "Attends a company-wide town hall meeting, listening to the CEO's speech.",
    "Takes a short walk during a break, getting some fresh air and exercise.",
    "Spends hours on a single document, trying to get the formatting perfect.",
    "Deals with a sudden change in project scope, requiring them to work overtime.",
    "Attends a workshop, learning a new skill that is relevant to their job.",
    "Spends hours on a single presentation, trying to make it visually appealing.",
    "Deals with a difficult client, trying to keep them happy and satisfied.",
    "Takes a sick day, hoping to recover from a bad cold.",
    "Works on a new marketing campaign, trying to come up with a catchy slogan.",
    "Spends hours on a single spreadsheet, trying to find a mistake.",
    "Deals with a sudden server crash, causing a major disruption to their work.",
    "Attends a networking event, meeting new people and expanding their professional circle.",
    "Takes a short break to call their parents or a loved one.",
    "Works on a new software feature, hoping to get it right on the first try.",
    "Spends hours on a single line of code, trying to find a bug.",
    "Deals with a sudden resignation of a team member, causing them to take on more work.",
    "Attends a hackathon, trying to come up with a new and innovative idea.",
    "Takes a short break to meditate or do some yoga, clearing their mind.",
    "Works on a new website, trying to make it user-friendly and visually appealing.",
    "Spends hours on a single design, trying to get the colors and fonts just right.",
    "Deals with a sudden change in management, causing them to adjust to a new leadership style.",
    "Attends a conference, listening to experts talk about the latest industry trends.",
    "Takes a short break to read a book or an article, learning something new.",
    "Works on a new mobile app, trying to make it fast and responsive.",
    "Spends hours on a single video, trying to get the editing just right.",
    "Deals with a sudden change in company policy, causing them to adjust their workflow.",
    "Attends a team-building retreat, spending a weekend with their colleagues.",
    "Takes a short break to play a video game or a game of chess.",
    "Works on a new product, trying to make it a success in the market.",
    "Spends hours on a single report, trying to make it concise and easy to read.",
    "Deals with a sudden change in project budget, requiring them to cut corners.",
    "Attends a company sports day, participating in a game of cricket or football.",
    "Takes a short break to listen to music or a podcast, relaxing their mind.",
    "Works on a new business proposal, trying to win a new client.",
    "Spends hours on a single pitch deck, trying to make it persuasive.",
    "Deals with a sudden change in project deadline, requiring them to work faster.",
    "Attends a company picnic, spending a day outdoors with their colleagues.",
    "Takes a short break to browse online stores, looking for a new gadget.",
    "Works on a new marketing strategy, trying to reach a wider audience.",
    "Spends hours on a single blog post, trying to make it informative and engaging.",
    "Deals with a sudden change in project requirements, causing them to re-do their work.",
    "Attends a company-wide charity event, giving back to the community.",
    "Takes a short break to chat with a colleague, catching up on personal life.",
    "Works on a new social media campaign, trying to go viral.",
    "Spends hours on a single email campaign, trying to get the open rate up.",
    "Deals with a sudden change in team members, causing them to train new people.",
    "Attends a company-wide hackathon, competing against other teams.",
    "Takes a short break to check their social media, liking and commenting on posts.",
    "Works on a new sales pitch, trying to close a new deal.",
    "Spends hours on a single sales report, trying to find a trend.",
    "Deals with a sudden change in project deliverables, causing them to work on a new feature.",
    "Attends a company-wide holiday party, celebrating the end of a successful year."
]

query = "What the employees has to deal with the most"

# Embed the documents and the query
doc_embeddings = embedding.embed_documents(documents)
query_embedding = embedding.embed_query(query)

# Calculate cosine similarity
scores = cosine_similarity([query_embedding], doc_embeddings)[0]

# Find the index of the highest similarity score
index = np.argmax(scores)
score = scores[index]

# Print the results
print(f"Query: {query}\n")
print(f"Most similar document: {documents[index]}\n")
print(f"Similarity score: {score}")