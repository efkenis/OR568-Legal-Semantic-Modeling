library(tidyverse)
library(tidytext)
library(caret)
library(tm)
library(xgboost)
library(nnet)


## Read in csv and add index column
BVA_classification <- read_csv("BVA_classifications.csv",
                               col_types = cols(rhetrole = col_factor())) %>%
    mutate(id = row_number()) %>%
    mutate_at(.vars = vars(rhetrole),
              .funs = forcats::fct_recode,
              Sentence = "Sentence",
              Finding = "FindingSentence",
              Evidence = "EvidenceSentence",
              Reasoning = "ReasoningSentence", 
              LegalRule = "LegalRuleSentence",
              Citation = "CitationSentence" 
              )


##-------------------------------------------------------------------------

## Bar chart of rhetroles
ggplot(BVA_classification, aes(x=fct_rev(fct_infreq(rhetrole)))) +
    geom_bar(stat="count", reverse=T) +
    coord_flip() +
    labs(x="Rhetorical Role", y="Counts",
         title="Counts of Rhetorical Roles in PTSD Claims Cases") +
    scale_y_continuous(breaks=seq(0,1200,200)) +
    theme(axis.title.x = element_text(size=11, vjust=0.5),
          axis.title.y = element_text(size=11),
          axis.text.x = element_text(size=10),
          axis.text.y = element_text(size=10),
          plot.title = element_text(size=12, hjust=0.5),
          panel.grid.minor = element_blank(),
          axis.ticks.x = element_blank(),
          axis.ticks.y = element_blank(),
          plot.margin = margin(15,15,15,15))

ggsave("barchart.png", width=6, height=3.5)


##-------------------------------------------------------------------------

## Count of all words sorted
BVAwords <- BVA_classification %>%
    unnest_tokens(word,sentences) %>%
    anti_join(stop_words, by="word")


## Bar plot of most common words without stop words
BVAwords %>%
    count(word, sort=T) %>%
    filter(n > 290) %>%
    mutate(word = reorder(word,n)) %>%
    ggplot(aes(word,n)) +
    geom_col(fill="#FF6666") +
    coord_flip() +
    labs(x = "Word", y = "Count", title="Most Common Words in Veteran's PTSD Claims Cases") +
    theme(axis.title.x = element_text(size=11, vjust=0.5),
          axis.title.y = element_text(size=11),
          axis.text.x = element_text(size=10),
          axis.text.y = element_text(size=10),
          plot.title = element_text(size=12, hjust=0.5),
          axis.ticks.x = element_blank(),
          axis.ticks.y = element_blank(),
          plot.margin = margin(15,15,15,15))

ggsave("word_count.png", width=6, height=3.5)

## Count of top words by rhetrole
BVAwords %>%
    count(rhetrole, word, sort=T) %>%
    bind_tf_idf(word, rhetrole, n) %>%
    group_by(rhetrole) %>%
    mutate(rank=dense_rank(desc(tf_idf))) %>%
    arrange(desc(tf_idf)) %>%
    group_by(rhetrole) %>%
    filter(row_number()<9) %>%
    ungroup() %>%
    mutate(word = reorder(word, tf_idf)) %>%
    ggplot(aes(word, tf_idf, fill=rhetrole)) +
    geom_col(show.legend = F) +
    facet_wrap(~rhetrole, scales="free") +
    coord_flip() +
    ggtitle("Words With Highest TF-IDF Per Rhetorical Role in Veteran's PTSD Claims Cases") + 
    theme(axis.title.x = element_text(size=10, vjust=0.6, hjust=0.4),
          axis.title.y = element_text(size=10),
          axis.text.x = element_text(size=7),
          axis.text.y = element_text(size=8),
          plot.title = element_text(size=10, hjust=0.5),
          panel.grid.minor = element_blank(),
          axis.ticks.x = element_blank(),
          axis.ticks.y = element_blank())

ggsave("facet_tf-idf1.png", width=7, height=4.5)



##-------------------------------------------------------------------------

# Cast dataframe to tf-idf matrix
sent <- BVAwords %>%
    count(word, id, sort=T) %>%
    cast_dtm(id, word, n, weighting=tm::weightTfIdf)

## sent <- removeSparseTerms(sent, sparse = 0.9995)

##Train-test split
indexes <- sample(seq(1,length(unique(BVAwords$id))), size=0.7*length(unique(BVAwords$id)))

trainsent <- as.matrix(sent)[indexes,]
testsent <- as.matrix(sent)[-indexes,]

## 4 sentences dropped because they only contain stop words
droppedsentences <-setdiff(unique(BVA_classification$id),unique(BVAwords$id))
rr <- BVA_classification$rhetrole[-droppedsentences]
rrnum <- as.numeric(rr) - 1

trainlabel <-rrnum[indexes]
testlabel <- rrnum[-indexes]


## Multinomial Logistic Regression
d <- data.frame(trainsent)
e <- data.frame(testsent)
logmodel <- multinom(trainlabel~.,data=d, maxit=12, MaxNWts=15000)

## Convert predicted labels to factor and create confusion matrix
predictedLabels <- as.factor(levels(rr)[max.col(fitted(logmodel))])
confusionMatrix(predictedLabels,
                rr[indexes],
                mode="everything")


## Predict labels for testing set
predmultinom <- predict(logmodel, newdata = e, type="probs")

predtest <- as.factor(levels(rr)[max.col(predmultinom)])

confusionMatrix(predtest,
                rr[-indexes],
                mode="everything")

