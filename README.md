# Predicting the Ideological Direction of Supreme Court Decisions: Ensemble of Justices vs. Unified Case-Based Model


## Research Question

Our goal was to predict the ideological direction ("liberal" or "conservative" as defined below) of US Supreme Court case decisions based mainly on the facts of the case as known before the Court rules or even hears arguments (e.g. area of the law it relates to, or lower-court outcomes as described below).  Parties interested in an upcoming case could obviously use this tool to assess their prospects for success and plan accordingly.  In litigation strategy, they could also use feature importances to assess whether a particular justice or constellation of justices has a particularly strong or atypical voting tendency when it comes to a particular issue (e.g. taxes, or civil rights).

An interesting methodological question we attempted to answer was whether it is more effective to model decisions of the Court as a whole (i.e. the single outcome of generally nine justices' votes), or to separately model decisions of individual justices in probability terms and then aggregate these probabilities to predict the outcome of the case (ensemble method).  Although we found the ensemble method not to add accuracy with past cases, it is a useful approach if the Court is changing or if we know a particular combination of justices will hear the case (say, a reduced eight-justice panel when Congress is unable to confirm a nominee).


## Data Source and Format

![SCDB_landing](/img/SCDB_landing.png)

We are deeply indebted to the source of our labeled data, http://supremecourtdatabase.org, which has complete, expertly coded and labeled data about US Supreme Court cases dating back to 1791.  We used the "modern" version of the database, with cases since 1946, since the Court functioned quite differently in earlier periods.  The modern database has two different versions, and we use both.  The case-centered database contains 8,893 cases (as rows), each of which is described in 53 columns.  The justice-centered database contains the same information, but each case is split into 5 to 9 rows, one for each justice who voted in the case (usually 9 of course, so there are 79,612 total rows).  The justice-centered database contains 8 extra columns with justice names and their votes, whether he or she wrote the majority opinion etc.

A small part of the raw data:
![raw_data](/img/raw_data.png)


## Target

We selected ideological direction (i.e. "liberal" vs. "conservative" as detailed on pages 50-52 of the codebook, included in this repo) as our target for prediction since this is generally what people have an intuition for and want to know.  Prior research by Katz, Bommarito and Blackman targeted case disposition, coded in the database as any of 11 categories but essentially indicating whether the Court decided to affirm or reverse a lower-court decision.  However, a significant number of decisions (15% of their dataset, which included pre-1946 cases), at least at the justice-centered level, were ambiguous.  Ideological direction, in contrast, is "unspecifiable" in only 1.7% of modern cases and blank for only 0.4%.  Ideological direction has the additional advantage of being balanced close to 50/50% between liberal and conservative.


## Features

As the lengthy codebook shows, all of the columns are essentially categorical.  In choosing predictor variables for our models, we first left aside non-predictive columns with unique values for each case (name of case, identification numbers, date of case).  Next we needed to eliminate any columns related to the outcome of a case.  This is because, in a genuine prediction scenario, we would not have data for anything not known prior to the Court's decision's being made public.  Thus we do not train our models with even tangentially outcome-related columns, such as the authority cited for a decision (could be 7 different values, including "statutory construction" and "federal common law") as this could contain some clue about the decision itself and would not be known prior to the case's hearing.

Please see the file features.xlsx for a summary of feature characteristics and which ones were deemed unknown before the case was decided (i.e "cheating" by containing information about the prediction target).

Possible values for an example feature, encoding the reason the Supreme Court agreed to hear a particular case:
![cat_vals_ex](/img/cat_vals_ex.png)
![var_cert_2](/img/var_cert_2.png)

Another concern in choosing predictors was the curse of dimensionality (multi-value columns leading to sparser arrays of dummy variables).  We have a reasonably large sample of cases, but not enough to support machine learning of generalizable distinctions among, for example, 310 different values of "petitioner type" (such as "state department or agency" vs. "state commission, board, committee or authority").  We attempted to balance the usefulness of any given variable against the number of values it could take, in the end settling on 14 variables that expand to around 450 columns once one-hot encoded.  As overfitting did later become an issue, future iterations of these models should narrow down/optimize the number of predictors further.


## Machine Learning Strategy and Results: Ensemble vs. Unified Model

Our initial strategy was the simpler of the two main approaches we took: we split the case-centered data into testing and training sets and tried various machine learning techniques to predict the ideological direction of the full Court's decisions.  Individual justice identities (except for who was chief justice at the time) and vote counts were not used.  Results were encouraging: 70% or so test accuracy (75% or so ROC-AUC), but we wanted to make use of the justice-centered data.

To do so we split the justice-centered into testing and training sets (by case so that justices on a particular case were either all in the testing set or all in training set).  We then split the training set by justice and trained models for each of the 37 justices, tuning hyper-parameters for each using `GridSearchCV`.  The justice-centered models had anywhere from 5,087 rows for Brennan to 82 rows for Gorsuch.  These could be used as-is if individual-justice decisions are of interest, but we decided to use them as an ensemble to predict full-Court decisions.

For each case, we assembled a vector of probabilities for the justices who voted on that case (again, usually 9 but as few as 5 historically).  These were the `predict_proba` outputs of individual justice models for the case in question.  Using a Poisson binomial distribution (since the probabilities are uneven), we found the probability that a majority of the justices' votes are liberal vs. conservative.  This was computed by adding up the Poisson-binomial probability mass function for, say, 5, 6, 7, 8 or 9 justices' voting liberal if there were 9 justices sitting.  The calculation was adjusted depending on how many justices were voting: the probability mass function was added for 4, 5 or 6 justices' voting liberal if only 6 justices were voting.

We expected the ensemble method to improve on the initial strategy since the individual justices models' were individually tuned; but for now, any gain in accuracy, ROC-AUC and so forth has been minimal.  We plan to investigate further.

Results for case_centered (unified) model:
![case_based](/img/case_based.png)

Part of a single-justice-based model's results:
![justice_based](/img/justice_based.png)

Results for ensemble-based model:                             
![ensemble_based](/img/ensemble_based.png)


## Note on Data Leakage

We had to be very careful to use the same test-train splits within each justice's model that we used for the case-centered approach.  That is, if a case was used for training in the case-centered approach, it had to be used only for training in each of the 9 or so justice-centered models where it was relevant.  Otherwise, if even one justice's model used the case for testing rather than training, the ensemble method would unfairly have "better" information about the test set than the case-centered strategy.  As a result, the test-train splits for each justice deviated randomly from the 70/30% split imposed on the case-centered data.  Another disadvantage of this necessary precaution was that it disallowed automatic k-folds cross validation as we needed to keep the same test-train split across all models.


## Interpreting Feature Importances

![RBG_feat_imps](/img/RBG_feat_imps.png)

As an example, the above feature importances for Justice Ginsburg were fairly typical.  Most important was "lcDispositionDirection."  This was the ideological direction of the lower court's decision ("1" is conservative, "2" is liberal, "3" is unspecifiable, and "999" is missing/NA).  This may be related to the fact that 60-70% of Supreme Court decisions since 1950 or so have been reversals (see page 10 of Katz et al.).  Katz et al. were able to show this because there is another variable in the database "caseDispositionSc" related to whether a case was reversed, affirmed, remanded etc., but we did not use it as a predictor because it has to do with the outcome of the case and would not be known a priori.  Since justices were largely reversing lower-court decisions, the model may have tended to predict the opposite ideological direction from that taken by the lower court.

The next most important feature was whether or not "decisionType" was "2," which corresponds to "per curiam (no oral argument)."  This may be because unanimous, uncontroversial decisions have allowed a justice to vote in an ideological direction she might not usually take.  Decision types "1" ("opinion of the court (orally argued)"), "6" ("per curiam (orally argued)") and "7" ("judgment of the Court (orally argued)") also influenced Ginsburg.

Next we found that whether or not the case had to do with "no merits: writ improperly granted" ("issue" number "90150") or "indigents: US Supreme Court docketing fee" ("issue" number "20360") also influenced Justice Ginsburg's decision.  Other variables like lawType ("2" is "constitutional amendment") could give further insight if examined by legal experts.


## Python Pipeline

Please see AutoGenerateFull.py for the main script.

First of course, we imported the databases, in both cases dropping rows where ideological direction is indeterminate.  We then dropped columns not used as predictors or target, one-hot encoded the remaining predictors and ran a battery of machine-learning methods for the case-centered data, for each justice individually and then the ensemble method.

Records were generated in a text file, showing, for each model run, test and train accuracy, ROC-AUC and log-loss.  For the test data, precision, recall and confusion matrix were recorded.  For individual-justice models, test and train accuracy, ROC-AUC and log-loss were shown both with default scikit-learn settings an after tuning.  The optimal values for hyper-parameters were also noted.  These optimal values varied quite a bit by justice, but in general trees needed to be kept shallow (maximum depth of around 10) to avoid overfitting.

These records were reproduced in a CSV output file.  For each model we also exported a CSV file with feature importances according to our random-forest classifier, which should be a rich source of further conclusions for this study.

Example of output:
![csv_meta](/img/csv_meta.png)


## Summary of Findings

By applying a range of supervised machine-learning techniques in scikit-learn - SVM, decision trees and their derivatives (random forests, xgboost, adaboost), KNN clustering and Naive Bayes - we predicted ideological outcome (liberal vs. conservative as defined below) of US Supreme Court cases with test accuracy (or out-of-bag score for bagged methods) of around 72% (vs. 50% null model).  Random forest algorithms tended to achieve the best results, though other techniques were close behind.  The exception was the extremely poor performance of Naive Bayes, which we attribute to the non-parametric distributions of our entirely categorical variables.  A preliminary assessment showed that a more complex ensemble method where multiple models were probability-voted to predict case outcomes was not necessarily more effective than simply training models with case-centered data alone.


## Citations

Harold J. Spaeth, Lee Epstein, et al. 2018 Supreme Court Database, Version 2018 Release 1. URL: http://Supremecourtdatabase.org

Katz DM, Bommarito MJ, II, Blackman J (2017) A general approach for predicting the behavior of the Supreme Court of the United States. PLoS ONE 12(4): e0174698. https://doi.org/10.1371/journal.pone.0174698

Poisson binomial Python library: https://github.com/tsakim/poibin
