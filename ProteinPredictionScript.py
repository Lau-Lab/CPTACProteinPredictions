# Edward Lau 2021
# This code uses the CPTAC package to download CPTAC data for machine learning.


import cptac

import pandas as pd
import re
import tqdm
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import ElasticNetCV

class LearnCPTAC(object):

    def __init__(self, cptac_df):

        from get_proteins import GetProtein, GetComplex

        self.df = cptac_df

        self.all_proteomics = [re.sub('_proteomics', "", protein) for protein in self.df.columns if
                               protein.endswith('_proteomics')]

        self.all_transcriptomics = [re.sub('_transcriptomics', "", transcript) for transcript in self.df.columns if
                                    transcript.endswith('_transcriptomics')]

        self.shared_proteins = [protein for protein in self.all_proteomics if protein in self.all_transcriptomics]

        self.tx_to_include = "self"
        self.train_method = "simple"

        self.stringdb = GetProtein()
        self.corumdb = GetComplex()

        pass

    def learn_all_proteins(
            self,
            tx_to_include="self",
            train_method="simple",
    ):
        """
        Wrapper for learning one protein

        :param tx_to_include: transcript to include (self, all, string, or corum)
        :param train_method: simple or voting
        :return:
        """

        self.tx_to_include = tx_to_include
        self.train_method = train_method

        learning_results = []

        for i, protein in enumerate(tqdm.tqdm(self.shared_proteins)):
            learning_result = self.learn_one_protein(protein)

            if learning_result:
                learning_results.append(learning_result)

                if i % 100 == 0:
                    corr_values = [metric[2].corr_test.values[0] for metric in learning_results]
                    r2_values = [metric[2].r2_test.values[0] for metric in learning_results]
                    nrmse = [metric[2].nrmse.values[0] for metric in learning_results]

                    tqdm.tqdm.write('{0}: {1}, r: {2}, R2: {3}, med.r: {4}, med.R2: {5}, med.NRMSE: {6}'.format(
                        i,
                        protein,
                        round(list(learning_result[2].corr_test.values)[0], 3),
                        round(list(learning_result[2].r2_test.values)[0], 3),
                        round(np.median([r for r in corr_values if not np.isnan(r)]), 3),
                        round(np.median([r2 for r2 in r2_values if not np.isnan(r2)]), 3),
                        round(np.median([nr for nr in nrmse if not np.isnan(nr)]), 3),
                    ))

        return learning_results

    def learn_one_protein(
            self,
            protein_to_do,
            returnModel=False,
    ):
        """

        :param protein_to_do:
        :param returnModel:  Return the model rather than the results, for examining coefficients
        :return:
        """

        y_df = self.df[[protein_to_do + '_proteomics']]
        y_df = y_df.dropna(subset=[protein_to_do + '_proteomics'])

        if self.tx_to_include == "self":
            proteins_to_include = [protein_to_do]

        else:
            raise Exception('tx to include is not self')

        # Join X and Y
        xy_df = self.df[[tx + '_transcriptomics' for tx in proteins_to_include]].join(y_df, how='inner').copy().dropna()

        # Skip proteins with fewer than 20 samples
        if len(xy_df) < 20:
            return []

        # Do train-test split
        x = xy_df.iloc[:, :-1]  # .values
        y = xy_df.iloc[:, -1]  # .values
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)
        # End

        if self.train_method == 'elastic':
            vreg = ElasticNetCV(l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1],
                                cv=5,
                                fit_intercept=False,
                                n_jobs=-1,
                                )

        else:
            raise Exception('training method is not elastic')

        vreg.fit(x_train, y_train)
        y_train_pred = vreg.predict(x_train)
        y_test_pred = vreg.predict(x_test)

        if returnModel:
            return (vreg)

        # Write train and test table
        x_train_out = x_train.copy()
        x_train_out['y_train'] = y_train
        x_train_out['y_train_pred'] = y_train_pred
        # X_train_out.to_csv(os.path.join(directory, protein_to_do + '_train.txt'))

        x_test_out = x_test.copy()
        x_test_out['y_test'] = y_test
        x_test_out['y_test_pred'] = y_test_pred
        # X_test_out.to_csv(os.path.join(directory, protein_to_do + '_test.txt'))

        # Mean square error and R2
        if np.std(y_test_pred) == 0:
            corr = 0
        else:
            corr = round(np.corrcoef(y_test, y_test_pred)[0][1], 4)
        r2_test = round(r2_score(y_test, y_test_pred), 4)
        nrmse = round(np.sqrt(mean_squared_error(y_test, y_test_pred) / (np.max(y_test) - np.min(y_test))), 4)
        metrics_df = pd.DataFrame(data={'corr_test': [corr],
                                        'r2_test': [r2_test],
                                        'num_proteins': [len(xy_df)],
                                        'nrmse': [nrmse],
                                        },

                                  index=[protein_to_do])

        return [round(x_train_out, 5), round(x_test_out, 5), round(metrics_df, 5)]

# Download CPTAC data
cptac.download(dataset="endometrial")
cptac.download(dataset="ovarian")
cptac.download(dataset="colon")
cptac.download(dataset="brca")
cptac.download(dataset="luad")

en = cptac.Endometrial()
ov = cptac.Ovarian()
co = cptac.Colon()
br = cptac.Brca()
lu = cptac.Luad()

# For endometrial, let's try getting the RNA and protein data
en_rna = en.get_transcriptomics()
en_pro = en.get_proteomics()
a = en.join_omics_to_omics('transcriptomics', 'proteomics')

ov_rna = ov.get_transcriptomics()
ov_pro = ov.get_proteomics()
b = ov.join_omics_to_omics('transcriptomics', 'proteomics')
b.columns = b.columns.droplevel(1)

co_rna = co.get_transcriptomics()
co_pro = co.get_proteomics()
c = co.join_omics_to_omics('transcriptomics', 'proteomics')

br_rna = br.get_transcriptomics()
br_pro = br.get_proteomics()
d = br.join_omics_to_omics('transcriptomics', 'proteomics')
d.columns = d.columns.droplevel(1)

lu_rna = lu.get_transcriptomics()
lu_pro = lu.get_proteomics()
e = br.join_omics_to_omics('transcriptomics', 'proteomics')
e.columns = e.columns.droplevel(1)

# StandardScaler Transform of the transcriptomics data (VST/log space) to match proteomics (standardized genewise) data.
a_std = a.copy()
a_tx_cols = [col for col in a_std.columns if col.endswith('transcriptomics')]
a_std[a_tx_cols] = StandardScaler().fit_transform(a_std[a_tx_cols])
a_std.index = 'EN' + a_std.index

b_std = b.copy()
b_std = b_std.loc[:, ~b_std.columns.duplicated(keep='first')]
b_tx_cols = [col for col in b_std.columns if col.endswith('transcriptomics')]
b_std[b_tx_cols] = StandardScaler().fit_transform(b_std[b_tx_cols])
b_std.index = 'OV' + b_std.index

c_std = c.copy()
c_tx_cols = [col for col in c_std.columns if col.endswith('transcriptomics')]
c_std[c_tx_cols] = StandardScaler().fit_transform(c_std[c_tx_cols])
c_std.index = 'CO' + c_std.index

d_std = d.copy()
d_std = d_std.loc[:, ~d_std.columns.duplicated(keep='first')]
d_tx_cols = [col for col in d_std.columns if col.endswith('transcriptomics')]
d_std[d_tx_cols] = StandardScaler().fit_transform(d_std[d_tx_cols])
d_std.index = 'BR' + d_std.index

e_std = e.copy()
e_std = e_std.loc[:, ~e_std.columns.duplicated(keep='first')]
e_tx_cols = [col for col in e_std.columns if col.endswith('transcriptomics')]
e_std[e_tx_cols] = StandardScaler().fit_transform(e_std[e_tx_cols])
e_std.index = 'LU' + e_std.index

z_df = pd.concat([a_std, b_std ,c_std, d_std, e_std])
comb = LearnCPTAC(z_df)

# Get the result from elastic net using just self transcript
self_elastic_result = comb.learn_all_proteins(tx_to_include="self", train_method="elastic")

metric = pd.concat([result[2] for result in self_elastic_result])
metric.to_csv("self_method-elastic_metrics.csv")
