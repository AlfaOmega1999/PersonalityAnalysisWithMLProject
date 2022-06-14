# Personality Analysis using ML

This project aims to create a free, open source application that uses machine learning techniques to determine the personality of an individual based on the text messages they write on social networks.

## MBTI
To determine what type of personality a person has, it is necessary to answer a series of questions from among four dichotomies where the person's preferences are very important and once the person answers these questions sincerely, we will obtain up to 16 different personality types that can be expressed as a four-letter code.


<img src="/images/Myers-Briggs Chart.webp" alt="MBTIPhoto"/>

## Frontend Deployment
[![Netlify Status](https://api.netlify.com/api/v1/badges/25f42b2a-f753-4543-92b7-be4af49d188b/deploy-status)](https://app.netlify.com/sites/personality-web/deploys)

https://personality-web.netlify.app/ 

## Backend Deployment
![Heroku](https://heroku-badge.herokuapp.com/?app=heroku-badge)

## Folder structure
### data
Contains the data used for the generation of the models.
### models
Contains the models used for personality detection.
### public
Contains part of the code used for React
### src
Contains the code for the personality detection application.

## Local Deployment
Frontend | Backend
`npm start` | `python -m uvicorn main:app --reload` 
