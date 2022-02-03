import pandas as pd


class Reader:
    df = pd.DataFrame()

    def read_data(self, folders=43, pre_path=r"./data/"):

        for i in range(0, folders):
            if i < 10:
                path = pre_path + str(i) + r"/GT-0000" + str(i) + ".csv"
            else:
                path = pre_path + str(i) + r"/GT-000" + str(i) + ".csv"

            df_tmp = pd.read_csv(path, sep=";")
            df_tmp.insert(1, "Folder", pre_path + str(i) + r"/")
            self.df = self.df.append(df_tmp)

        self.df.reset_index(inplace=True)
        self.df.drop("index", axis=1, inplace=True)
        return self.df
    