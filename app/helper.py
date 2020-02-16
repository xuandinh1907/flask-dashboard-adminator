# -*- encoding: utf-8 -*-
"""
Flask Boilerplate
Author: AppSeed.us - App Generator 
"""

# from flask   import json, url_for, jsonify, render_template
# from jinja2  import TemplateNotFound
# from app     import app

# from . models import User
# from app    import app,db,bc,mail
# from . common import *
# from sqlalchemy import desc,or_
# import hashlib
# from flask_mail  import Message
# import re
# from flask       import render_template

# import      os, datetime, time, random

# # build a Json response
# def response( data ):
#     return app.response_class( response=json.dumps(data),
#                                status=200,
#                                mimetype='application/json' )

# def g_db_commit( ):

#     db.session.commit( );    

# def g_db_add( obj ):

#     if obj:
#         db.session.add ( obj )

# def g_db_del( obj ):

#     if obj:
#         db.session.delete ( obj )

def get_add_tokens(do_enumerate):
    tags = ['Dd', 'Dl', 'Dt', 'H1', 'H2', 'H3', 'Li', 'Ol', 'P', 'Table', 'Td', 'Th', 'Tr', 'Ul']
    opening_tags = [f'<{tag}>' for tag in tags]
    closing_tags = [f'</{tag}>' for tag in tags]
    added_tags = opening_tags + closing_tags
    # See `nq_to_sqaud.py` for special-tokens
    special_tokens = ['<P>', '<Table>']
    if do_enumerate:
        for special_token in special_tokens:
            for j in range(11):
              added_tags.append(f'<{special_token[1: -1]}{j}>')

    add_tokens = ['Td_colspan', 'Th_colspan', '``', '\'\'', '--']
    add_tokens = add_tokens + added_tags
    return add_tokens
