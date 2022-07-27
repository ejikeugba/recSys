import numpy as np
import pandas as pd
import difflib_s
import streamlit as st


class recEngine_py():
    def __init__(self):
        self.userID = None
        self.book_df = None
        self.smatrix = None
        self.recommendation_engine = None

    def bookTracer(self, book_df, title, singleUse=True):
        self.book_df = book_df
        self.title = title
        self.singleUse = singleUse

        title = title.casefold()
        bk = book_df
        bk["titleLow"] = bk["title"].str.lower()
        bk_df = bk[bk.eq(title).any(axis=1)]

        if singleUse:
            bk_df = bk_df.drop(["titleLow"], axis=1)
        else:
            bk_df = bk_df.reset_index().drop(
                ["titleLow", "index", "userID"], axis=1)

        return bk_df

    def RecEng(self, userID, book_df, smatrix, noBooks=5):

        self.userID = userID
        self.book_df = book_df
        self.smatrix = smatrix

        user_list = book_df[book_df["userID"] == userID]
        list_of_all_titles = book_df["title"].tolist()
        book_name = user_list.iloc[0, ]["title"].lower()

        find_close_match = difflib_s.get_close_matches(
            book_name, list_of_all_titles)

        if len(find_close_match) > 0:
            close_match = find_close_match[0]
            index_of_the_book = self.bookTracer(book_df, close_match, singleUse=True)[
                'userID'].values[0]

            similarity_score = list(enumerate(smatrix[index_of_the_book-1]))
            sorted_similar_books = sorted(
                similarity_score, key=lambda x: x[1], reverse=True
            )[1:]

            origBook = self.bookTracer(book_df, close_match, singleUse=False)

            recomList = []

            for book in sorted_similar_books:
                index = book[0]
                if len(recomList) < noBooks:
                    zz = book_df[book_df.index == index]["title"].values

                    if len(zz) != 0:
                        title_from_index = zz[0]
                    else:
                        title_from_index = ""
                    if title_from_index:
                        recomList.append(self.bookTracer(
                            book_df, title_from_index, singleUse=False))
                        recL = pd.DataFrame(np.concatenate(recomList))
                        zz = recL.shape[0]
                        recL = pd.DataFrame(
                            np.insert(recL.values, zz, values=origBook, axis=0)
                        )
                        recL.columns = book_df.columns[1:8]

            return recL  # recEng

        else:
            return None
