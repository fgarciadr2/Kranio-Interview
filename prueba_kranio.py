#!/usr/bin/env python
# coding: utf-8

# # Preparing the data

# ## Import packages

# In[1]:


import datetime
import calendar
import pandas
import pyspark
from pyspark.sql.functions import *
from pyspark.sql import SparkSession
from pyspark.sql import Window
from pyspark.sql.types import IntegerType

import plotly.express as px

import numpy as np


# ## Start Spark session

# In[2]:


spark = SparkSession \
    .builder \
    .appName("Kranio Interview") \
    .getOrCreate()

spark.conf.set("spark.sql.legacy.timeParserPolicy","LEGACY") 


# ## Import database

# In[3]:


df = spark.read.format("csv").option("header", "true").load("C:/Users/felip/OneDrive - Microsoft/Prueba Kranio/movies.csv")


# ## Remove unused column

# In[4]:


df = df.drop("overview", "tagline", "backdrop_path", "poster_path", "keywords", "credits")

df = df.select("id","title","genres","release_date","budget","revenue", "popularity")

df = df.distinct()


# ## Create new columns 

# In[5]:


df = df.withColumn("profit", df.revenue-df.budget)
df = df.withColumn("release_date", to_date("release_date", "yyyy-MM-dd"))
df = df.withColumn("year", year("release_date"))
df = df.withColumn("month", month("release_date"))
df = df.withColumn("semester", when(df.month >= 6, 2).
                    otherwise(1))

df = df.filter(df.year <= 2022)


# In[6]:


# Tables to write the metrics


# ## Metric 1: Movies by semester

# In[7]:


#Cleaning data: Drop Null Values for "profit" and "semester". Also drop when revenue equals 1.000.000.000 d
df_semester = df.filter(df.profit > 0)
df_semester = df_semester.filter(df_semester.profit.isNotNull())
df_semester = df_semester.filter(df_semester.release_date.isNotNull())
df_semester = df_semester.filter(df_semester.revenue != 1000000000.0)
df_semester = df_semester.filter(df_semester.profit != 0)

# Create a window to sort the data by semester and profit, allowing
window_sem= Window.partitionBy(df_semester['semester']).orderBy(df_semester['profit'].desc())

#Create the rank variable
table_semester_profit = df_semester.select('*', rank().over(window_sem).alias('rank'))

#Show result example
table_semester_profit.show(5)


# This table has the column rank, which shows the profits' ranking for every movie, according to the semester.

# ## Metric 2: Genres by year and profits

# In[10]:


#Cleaning data: Drop null values and movies released after the current year (2022). Also dropping movies with values 0 and 1.000.000.000, which are used for data with no information

df_genres = df.filter(df.profit.isNotNull())
df_genres = df_genres.filter(df_genres.genres.isNotNull())
df_genres = df_genres.filter(df_genres.revenue != 1000000000.0)
df_genres = df.filter(df_genres.year.isNotNull())
df_genres = df_genres.filter(df_genres.year <= 2022)
df_genres = df_genres.filter(df_genres.profit != 0)

#Counting the number of times that the character "-" appears at each cell. Allowing us to know how many genres each movie has.
df_genres = df_genres.withColumn('number_genres', size(split(col("genres"), r"\-")))

#Separating the column gender by  
split_col = pyspark.sql.functions.split(df['genres'], '-')

#Calculate the movie with nost genres
row1 = df_genres.agg({"number_genres": "max"}).collect()[0]
m = row1["max(number_genres)"]

# Naming every new column concatenating "genre_" with its respective number from one to the maximum of genres for a movie

for i in range(0,5):
    NOM = "genre_" + str(i + 1)
    df_genres = df_genres.withColumn(NOM, split_col.getItem(i))

#Create a data frame where each column is a genre, creating rows for each gender of movies. That means that instead of each row representing a movie, each row represents a genre.      

for i in range(1, 6):
    NOM = "genre_" + str(i)
    if i == 1:
        df_genres2 = df_genres.select("id", "title", "genres", "release_date", "budget", "revenue",
                "profit", "year","number_genres", NOM).withColumnRenamed(NOM, "genre")
        df_genres2 = df_genres2.filter(df_genres2.genre.isNotNull())
    else:
        df_genres2_b = df_genres.select("id", "title", "genres", "release_date", "budget", "revenue",
                "profit", "year","number_genres", NOM).withColumnRenamed(NOM, "genre")
        df_genres2_b = df_genres2_b.filter(df_genres2_b.genre.isNotNull())
        df_genres2 = df_genres2.union(df_genres2_b)
                                      
#Create a data frame whith de sum of the profits for each gender at each year
            
df_genres_year = df_genres2.groupBy("genre", "year").agg(sum("profit").alias("profit"))

#Creates a window sorting movies by year and profit
window_genre = Window.partitionBy(df_genres_year['year']).orderBy(df_genres_year['profit'].desc())

#Create a ranking of the profits for each genre
table_genre_year = df_genres_year.select('*', rank().over(window_genre).alias('rank')).filter(col('rank') <= 5)

table_genre_year.show(5)


# This table contains the ranking for the profits of every gender by year. In order to be able to do that, I used loops to create a data frame where each rows represents a genre. That data frame is also used for other solutions of this test. 

# ## Metric 3: Most profitable movies by genre

# In[11]:


#Create a window sorting the data frame of genres for profit and genre
window_genre= Window.partitionBy(df_genres2['genre']).orderBy(df_genres2['profit'].desc())

#Sort genres by profits
table_genre_year.groupBy("genre").agg(sum(table_genre_year.profit).alias("profit")).sort(col("profit").desc())

#List of the five most profitable genres
list_genres = ["Adventure", "Action", "Comedy", "Drama", "Fantasy"]

#Create a data frame with the 10 most profitable movies by genre
table_genre_profit = df_genres2.select('*', rank().over(window_genre).alias('rank')).filter(col('rank') <= 10).filter(col("genre").isin(list_genres))

table_genre_profit.show(5)


# This table contains the 10 most profitable movies for each of the five most profitable genres 

# ## Metric 4: Popularity by month

# In[12]:


#Converting the column "popularity" from string to integer.
df = df.withColumn("popularity",df.popularity.cast(IntegerType()))

#Cleaning data, droping movies where values for popularity and month are not NULL
df_popularity = df.filter(df.popularity.isNotNull())
df_popularity = df_popularity.filter(df.month.isNotNull())

#Create a table with the mean of the popularity for every movie
table_popularity = df_popularity.groupBy("month").agg(mean("popularity").alias("popularity"))

table_popularity.show(5)


# In[13]:


## This table contains the average popularity of movies for each release month


# ## Metric 5: Number of movies by genre

# In[14]:


#create an object for the current year
current_year = datetime.date.today().year

#Use the data frame where each row represents a gere, selecting only rows where the movie was released at the current year or at one of the four previous years
df_five = df_genres2.filter(df_genres2.year <= current_year)
df_five = df_five.filter(df_five.year > current_year- 4)

#Create a data frame with the number of movies for each genre and each year.
table_releases = df_five.groupBy("year", "genre").count()

#Create a data frame with the budget of each genre at the last five years (This is not asked, but it complements the solution, by giving a posible explanation.)
table_releases_cost = df_five.groupBy("genre").agg(sum("budget").alias("budget"),
                                                  count("genre").alias("releases"))

table_releases_cost.show(5)


# This table contains the number of movies released and the budget for each genre. The budget is used to compliment the data, testing if a possible reason for genres to be less released is their cost.  

# # Loading data frames to Panda

# In[15]:


t1 = table_semester_profit.toPandas()
t2 = table_genre_year.toPandas()
t3 = table_genre_profit.toPandas()
t4 = table_popularity.toPandas()
t5 = table_releases.toPandas()


# # Show metrics

# ##Metric 1

# In[16]:


t1['semester'] = t1['semester'].astype(str)

fig1 = px.scatter(t1, x = "profit", y="profit", color ="semester", hover_data = ["title"],
                 title = 'Movies profits by semester',
                 labels={
                "semester": "Semester",  "profit": "Profit",

            },
                  template="simple_white"
)

fig1


# The most profitable movies at first semesters are Avengers: Endgame, Avengers: Infinity Wars and Fast and Furious 7.
# On the other hand, The most profitable movies ar second semesters are Avatar, Titanic, Star Wars: The Force Awakens, Spider-Man: No way Home, Jurasic World, Top Gun: Maverick, and Harry Potter and the Deathly Hallows part 2.
# Therefore, in the case of exceptionally highly profitable movies, most of them were released at the second semester. The next plot compares the distribution of the profits by semester.

# In[17]:


px.histogram(t1, x = "profit", color = "semester",
                 title = 'Movies profits by semester',
                 labels={
                "semester": "Semester",  "profit": "Profit",

            },
                  template="simple_white")


# The histogram sugest that movies released on second semesters are equally profitable than movies made at the first semesters. It also shows that more movies are released at second semesters. However, this plot is hard to read, because most movies have low profits, therefore, it is better to normalize the data using a logarithmic scale.

# In[18]:


t1['lprofit'] = np.log(t1['profit'])

px.histogram(t1, x = "lprofit", color = "semester",
                 title = 'Movies profits by semester',
                 labels={
                "semester": "Semester",  "profit": "Profit"},
                  template="simple_white")


# With this data is easier to see that most movies were released at second semester, but the profits has similar distribution. Therefore, the first plot shows that probably, exceptionally profitable movies where made at second semesters, simply because more movies are made on that period. 

# ## Metric 2

# In[19]:


t2 = table_genre_year.toPandas()

fig = px.line(t2, x='year', y='profit',
              facet_col='genre', facet_col_wrap = 6,
              template="simple_white",
             labels = {"profit":"Profit",
                      "genre":"Genre",
                      "year":"Year"})
fig.show()


# For most genders, movies started to get most profits around the early 80's. In the case of family movies, they started to get more profits at the 60's. It is important to mention that the profits are not adjusted by inflation. Another interesting thing is that by 2020, profits started to drop, due to the COVID-19 pandemic.

# ## Metric 3

# In[20]:


t3 = table_genre_profit.toPandas()

fig3 = px.bar(t3, x = "genre", y = "profit", color = "genre", text = "title",
             template = "simple_white",
              labels = {"profit":"Profit", "genre":"Genre"}
             )

fig3.show()


#  The ten movies most profitable by gender, are recent movies, because profits are not adjusted by inflation. A big part of the profits for the genres "Adventure", "Action" and "Fantasy" are explained by Avatar, which has those three genres at the same time. Avengers: Endgame are also a big explanation for the high profits of "Adventure" and "Action".

# ## Metric 4

# In[43]:


t4 = table_popularity.toPandas()
t4 = t4.sort_values(by=['month'])
t4['month_str'] = t4['month'].apply(lambda x: calendar.month_abbr[x])

fig4 = px.bar(t4, x = "month_str", y = "popularity", color = "popularity",
                        template = "simple_white",
              labels = {"popularity": "Popularity",
                    "month_str" : "Month"},
                          color_continuous_scale = ["yellow", "green", "black"]
)

fig4.show()


# January is a bad month to release a movie, because of the low popularity of the movies released that month. While movies released in december are the most popular, the months between February and November are good months too, especialy from June.

# ## Metric 5

# In[37]:


t5 = table_releases.toPandas()
t5['year'] = t5['year'].apply(str)
t5 = t5.sort_values(by = ["count"], ascending = False)
fig5 = px.bar(t5, x = "genre", y = "count", color = "year",             
              template = "simple_white",
              labels = {"count" : "Releases",
                       "genre" : "Genre",
                       "year" : "Year"}
             )

fig5.show()


# Drama is the genre with most releases in the last five years, followed by Comedy, Thriller, Documentary and Horror. While previous plots sugest that adventure and action are more profitable gender than those, they don't have so many releases, probably because they are most expensive to make, as the next plot suggest.

# In[44]:


t5_b = table_releases_cost.toPandas()
t5_b = t5_b.sort_values(by = ["budget"], ascending = False)

fig5 = px.bar(t5_b, x = "genre", y = "budget", color = "budget",
              template = "simple_white",
              labels = {"budget" : "Budget",
                       "genre" : "Genre",
                       "year" : "Year"},
             color_continuous_scale = ["yellow", "green", "black"])
fig5.show()


# Indeed, Drama, Comedy and Thriller are cheaper genres than Adventure and Action. That might be a reason why these to have less releases, even when they are mor profitable. 

# In[45]:


load_ext pycodestyle_magic

