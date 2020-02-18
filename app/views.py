# -*- encoding: utf-8 -*-
"""
License: MIT
Copyright (c) 2019 - present AppSeed.us
"""

# Python modules
import os, logging 

# Flask modules
from flask               import render_template, request, url_for, redirect, send_from_directory
from flask_login         import login_user, logout_user, current_user, login_required
from werkzeug.exceptions import HTTPException, NotFound, abort

# App modules
from app        import app, lm, db, bc
from app.models import User
from app.forms  import LoginForm, RegisterForm
from .convert_to_crops import *

import tensorflow as tf
import requests
import json
from flask import jsonify
from flask import request
from .models import model , tokenizer

# provide login manager with load_user callback
@lm.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Logout user
@app.route('/logout.html')
def logout():
    logout_user()
    return redirect(url_for('index'))

# Register a new user
@app.route('/register.html', methods=['GET', 'POST'])
def register():
    
    # declare the Registration Form
    form = RegisterForm(request.form)

    msg = None

    if request.method == 'GET': 

        return render_template( 'pages/register.html', form=form, msg=msg )

    # check if both http method is POST and form is valid on submit
    if form.validate_on_submit():

        # assign form data to variables
        username = request.form.get('username', '', type=str)
        password = request.form.get('password', '', type=str) 
        email    = request.form.get('email'   , '', type=str) 

        # filter User out of database through username
        user = User.query.filter_by(user=username).first()

        # filter User out of database through username
        user_by_email = User.query.filter_by(email=email).first()

        if user or user_by_email:
            msg = 'Error: User exists!'
        
        else:         

            pw_hash = password #bc.generate_password_hash(password)

            user = User(username, email, pw_hash)

            user.save()

            msg = 'User created, please <a href="' + url_for('login') + '">login</a>'     

    else:
        msg = 'Input error'     

    return render_template( 'pages/register.html', form=form, msg=msg )

# Authenticate user
@app.route('/login.html', methods=['GET', 'POST'])
def login():
    
    # Declare the login form
    form = LoginForm(request.form)

    # Flask message injected into the page, in case of any errors
    msg = None

    # check if both http method is POST and form is valid on submit
    if form.validate_on_submit():

        # assign form data to variables
        username = request.form.get('username', '', type=str)
        password = request.form.get('password', '', type=str) 

        # filter User out of database through username
        user = User.query.filter_by(user=username).first()

        if user:
            
            #if bc.check_password_hash(user.password, password):
            if user.password == password:
                login_user(user)
                return redirect(url_for('index'))
            else:
                msg = "Wrong password. Please try again."
        else:
            msg = "Unknown user"

    return render_template( 'pages/login.html', form=form, msg=msg ) 

# App main route + generic routing
@app.route('/', defaults={'path': 'index.html'})
@app.route('/<path>')
def index(path):

    if not current_user.is_authenticated:
        return redirect(url_for('login'))

    content = None

    try:

        # @WIP to fix this
        # Temporary solution to solve the dependencies
        if path.endswith(('.png', '.svg' '.ttf', '.xml', '.ico', '.woff', '.woff2')):
            return send_from_directory(os.path.join(app.root_path, 'static'), path)    

        # try to match the pages defined in -> pages/<input file>
        return render_template( 'pages/'+path )

    except:
        
        return render_template('layouts/auth-default.html',
                                content=render_template( 'pages/404.html' ) )


@app.route('/qa_processing', methods=['POST','GET'])
def get_data_api():

    data = json.loads(request.data)
    paragraph = data.get("para",None)
    questions = data.get("ques",None)
    # paragraph="Do you know what is the difference between you and stars ? \nThe stars are on the sky and you are in my heart !"
    # questions=["what is the difference between you and stars"]
    print(paragraph)
    print(questions)
    print(type(paragraph))
    print(type(questions))
    qa = {}
    questions = questions.strip().split("\n")
    for question in questions:
        question_tokens = tokenizer.tokenize(question)
        paragraph_tokens = tokenizer.tokenize(paragraph)
        tokens = ['[CLS]'] + question_tokens + ['[SEP]'] + paragraph_tokens + ['[SEP]']
        input_word_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_word_ids)
        input_type_ids = [0] * (1 + len(question_tokens) + 1) + [1] * (len(paragraph_tokens) + 1)

        input_word_ids, input_mask, input_type_ids = map(lambda t: tf.expand_dims(
            tf.convert_to_tensor(t, dtype=tf.int32), 0), (input_word_ids, input_mask, input_type_ids))
        outputs = model([input_word_ids, input_mask, input_type_ids])
        # using `[1:]` will enforce an answer. `outputs[:][0][0]` is the ignored '[CLS]' token logit.
        short_start = tf.argmax(outputs[0][0][1:]) + 1
        short_end = tf.argmax(outputs[1][0][1:]) + 1
        answer_tokens = tokens[short_start: short_end + 1]
        answer = tokenizer.convert_tokens_to_string(answer_tokens)
        qa[question] = answer
        
    print(qa)
    # return {"Test":"Test"}
    return jsonify(qa)

@app.route('/qa_link_processing', methods=['POST','GET'])
def get_data_link_api():

    data = json.loads(request.data)
    link = data.get("wiki",None)
    questions = data.get("ques",None)
    # paragraph="Do you know what is the difference between you and stars ? \nThe stars are on the sky and you are in my heart !"
    # questions=["what is the difference between you and stars"]
    questions = questions.strip().split("\n")
    print(link)
    print(questions)
    print(type(link))
    print(type(questions))
    my_squad = demo(link,questions)
    return my_squad
    #return jsonify(qa)