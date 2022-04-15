import json 
import os

with open("bump.json","rb") as f:
    bump_json = json.load(f)
    
os.environ['BUMP'] = bump_json['bump_rule']
