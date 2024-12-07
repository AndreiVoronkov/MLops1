# from flask import Flask
# from flask_restx import Api, Resource, fields

# app = Flask(__name__)
# api = Api(app)

# passwords = []
# a_password = api.model('Resource', {'password': fields.String})

# @api.route('/password')
# class Prediction(Resource):
#     def get(self):
#         return passwords
    
#     @api.expect(a_password)
#     def post(self):
#         passwords.append(api.payload)
#         return {'Result': 'pass added'}, 201
    
    
    