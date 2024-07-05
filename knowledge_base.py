from fuzzywuzzy import fuzz
from fuzzywuzzy import process
from pymongo import MongoClient
import json


class KnowledgeBase:
    def __init__(self, connection_string, db_name="Cluster0",collection_name = "knowledge_base"):
        self.client = MongoClient(connection_string)
        self.db = self.client[db_name]
        self.collection = self.db[collection_name]

    def initialize_json(self,json_file_path):
        with open(json_file_path, 'r') as file:
            initial_data = json.load(file)
        
        for topic,content in initial_data.items():
            self.add_entry(topic,content)
        
        print(f"Initialized database with {len(initial_data)} entries")
    

    def add_entry(self,topic, content):
        self.collection.update_one(
            {"topic": topic.lower()},
            {"$set": {"content" : content}},
            upsert=True
        )

    def get_all_topics(self):
        return [doc["topic"] for doc in self.collection.find({}, {"topic" : 1})]
    
    def get_diabetes_info(self,query):
        topics = self.get_all_topics()
        best_match = process.extractOne(query.lower(), topics,score_cutoff=60)
        if best_match:
            result = self.collection.find_one({"topic": best_match[0]})
            return result["content"] if result else None
        else: 
            return "I'm sorry, I don't have specific information about that. Please try asking a different question."

    def close(self):
        self.client.close()

if __name__ == "__main__":
    connection_string = "mongodb+srv://rabirabi:Rabirabi80@cluster0.ylk5353.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

    kb = KnowledgeBase(connection_string)

    kb.initialize_json('diabetes_knowledge_base.json')

    info = kb.get_diabetes_info("What are the symptoms of diabetes?")
    print(info)

    kb.close()
