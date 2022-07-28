import urllib.request
import streamlit as st
import pandas as pd
from PIL import Image
from recEngine import recEngine_py
from itertools import cycle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from stop_words import get_stop_words


s = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Atma:wght@600&display=swap')
"""

st.markdown(s, unsafe_allow_html=True)


def newStyle(param, title=True):
    if title:
        new_thm = '<p style="font-family:Atma; color:rgb(255, 121, 3); font-size: 42px;">' + \
            param + '</p>'
        return new_thm
    else:
        new_thm = '<p style="font-family:Sherif; color:rgb(255, 121, 3); font-size: 20px;">' + \
            param + '</p>'
        return new_thm


st.markdown(newStyle('Book-Crossing Recommender Engine'),
            unsafe_allow_html=True)


# with st.container():
#    background = Image.open('Book.jpg')
#    st.image(background, width=700)
st.image('Book.jpg')

st.text("")

st.markdown(newStyle('About', title=False),
            unsafe_allow_html=True)
st.markdown("""
This app conducts a content-based recommendation of books in the book-crossing 
platform. Select your favorite book and get interesting recommendations of similar books 
to read - it's that simple!
""")

st.text("")

st.markdown(newStyle('Recommended Books', title=False),
            unsafe_allow_html=True)

DFurl = "https://raw.githubusercontent.com/ejikeugba/Statics/main/data/"


# @st.cache(allow_output_mutation=True)
def load_df(path):
    books = pd.read_csv(path+"BX-data.csv", sep=",",
                        on_bad_lines="skip", encoding="latin-1")
    return books.head(5000)


book_df = load_df(DFurl)


# @st.cache()
def cosine_sim(df_var):
    combined_features = (
        df_var["title"] + " " + df_var["author"]
    )
    stopwords_list = (
        get_stop_words("english") + get_stop_words("french") +
        get_stop_words("german"))

    vectorizer = TfidfVectorizer(
        stop_words=stopwords_list, lowercase=True, strip_accents='unicode', use_idf=True)
    feature_vectors = vectorizer.fit_transform(combined_features)
    similarity = cosine_similarity(feature_vectors)

    return similarity


smatrix = cosine_sim(book_df)
list_of_all_titles = book_df["title"].tolist()

re = recEngine_py()

st.sidebar.info('**Select your favorite book:**')
option = st.sidebar.selectbox(
    '',
    (list_of_all_titles), index=743)  # 10486

userInput = re.bookTracer(book_df, option, singleUse=True)

urlx = userInput['imgUrl'].values[0]
captx = userInput['title'].values[0]

try:
    urllib.request.urlretrieve(urlx, 'imgx.jpg')
except Exception as exc:
    print(
        f"Exception occured while downloading image from url {urlx} {str(exc)}")

st.sidebar.image('imgx.jpg', width=150, caption=captx)

with st.sidebar.expander("view book info"):
    st.write("**ISBN:** ", userInput.ISBN.values[0])
    st.write("**Author:** ", userInput.author.values[0])
    st.write("**Publisher:** ", userInput.publisher.values[0])

st.sidebar.text("")
st.sidebar.text("")

st.sidebar.info('**Number of books to recommend:**')
number_of_books = st.sidebar.slider(
    '', 1, 20, 8)

userID = userInput.userID.values[0]
bkrc = re.RecEng(userID, book_df, smatrix, noBooks=number_of_books)


if (bkrc is not None):

    ans = bkrc.iloc[:-1]
    imgs = ans['imgUrl']
    caption = ans['title']
    cols = cycle(st.columns(4))

    url_list = imgs
    filename = 1

    for url in url_list:
        try:
            urllib.request.urlretrieve(url, f'{filename}.jpg')
            filename += 1
        except Exception as exc:
            print(
                f"Exception occured while downloading image from url {url} {str(exc)}")

    imgList = []
    for x in range(1, number_of_books+1):
        imgList.append(str(x)+'.jpg')

    for idx, img in enumerate(imgList):
        next(cols).image(img, width=150, caption=caption[idx])

    st.text("")

    finalDF = ans.drop(['imgUrl', 'rating'], axis=1)
    finalDF.index += 1

    st.markdown(newStyle('Book Info', title=False),
                unsafe_allow_html=True)
    st.write(finalDF)

    @ st.cache
    def convert_df(df):
        return df.to_csv().encode('utf-8')

    csv = convert_df(finalDF)

    st.download_button(
        label="Download book info as CSV",
        data=csv,
        file_name='recommended_books.csv',
        mime='text/csv',
    )
else:
    st.info(
        '**No close match currently found for the selected book! Please make a different selection.**')

st.text("")
st.text("")

st.markdown(newStyle('Data source', title=False),
            unsafe_allow_html=True)
st.markdown("""
The data used in this project was adapted from the book-crossing dataset available [here](http://www2.informatik.uni-freiburg.de/~cziegler/BX/). 
The data was originally collected by Cai-Nicolas Ziegler in a 4-week crawl (August / September 2004) from the Book-Crossing community. 
It contains a total of 278,858 users (anonymized but with demographic information) providing 1,149,780 ratings (explicit / implicit) and 
about 271,379 books. For further details on the dataset see the publication [here](https://dl.acm.org/doi/10.1145/1060745.1060754). 
The figures below show the geographical locations of users in the book-crossing dataset.
""")

st.image("geoRegion.png")

st.text("")
st.text("")

st.markdown(newStyle('Related Projects', title=False),
            unsafe_allow_html=True)
st.markdown("""
Further analysis of the dataset and related projects are available at [DataXotic](https://ejikeugba.github.io/DataXotic/project/).
""")

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
