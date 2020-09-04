from flask import Flask,request, jsonify
from flask_restful import Resource, Api
import pickle
import numpy as np

# Init app
app = Flask(__name__)
api = Api(app)

model = pickle.load(open('model.pickle', 'rb'))

# Endpoint Class
class Root(Resource):
   def get(self):  
        """
        root endpoint of the api
        """        
        return jsonify({"endpoint1" : "i am the root",
                "endpoint2" : "/predict/"
               
            })



class predict(Resource):
    def get(self):
        return jsonify({'endpoint task' : 'takes the feature values and predicts sepsis <br> the list of features: [Patient_id, HR, Temp, SBP, MAP, DBP, RESP, GENDER, ICULUS]'})
    
    def post(self):
        json_post = request.get_json()

        HR = int(json_post['HR'])
        TEMP = int(json_post['TEMP'])
        SBP = int(json_post['SBP'])
        MAP = int(json_post['MAP'])
        DBP = int(json_post['DBP'])
        RESP = int(json_post['RESP'])
        GENDER = int(json_post['GENDER'])
        ICULUS = int(json_post['ICULUS'])

        test = (HR, TEMP, SBP, MAP, DBP, RESP, GENDER, ICULUS)
        test = np.asarray(test).reshape(1, -1)


        if model.predict(test) == 1:
            response = {'Sepsis' : 'positive'}
        else:
            response = {'Sepsis' : 'negative'}

        return jsonify(response)


# Api endpoints
api.add_resource(Root,'/')
api.add_resource(predict, '/predict/')



# Run Server
if __name__ == '__main__':
    #creating a server
    #set debug=False on production
    app.run(host='0.0.0.0', debug=False)

