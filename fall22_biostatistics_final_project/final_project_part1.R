### BIOL 5312 final project part 1

library(tidyverse)
library(car)
library(ggplot2)
library(ggpubr)
library(rstatix)

## retrieve data
# setwd("C:/Users/tuf61393/OneDrive - Temple University/BINFO PSM/Semester3_Fall2022/biol5312bst/fp")
cpp<-read.csv("2021CPPstudy.csv")
dim(cpp)
headcpp<-head(cpp)

# confirm criteria
cpp<-filter(cpp,CT>=100,ET<=100,startsWith(Injection,"hit"))[,c(1,3,5:7)]
dim(cpp)
# number treatments for ordering
cpp$Treatment[cpp$Treatment=="oCRF"]<-"1. oCRF"
cpp$Treatment[cpp$Treatment=="vehicle"]<-"2. vehicle"
cpp$Treatment[cpp$Treatment=="NBI+swim"]<-"3. NBI+swim"
cpp$Treatment[cpp$Treatment=="vehicle+swim"]<-"4. vehicle+swim"
# reformat ID
for (i in c(1:nrow(cpp))){
  cpp$ID[i]<-toString(cpp$ID[i])
}
cpp$ID[37]<-"9.10"

# convert to long-format dataframe
cpp<-cpp %>% gather(key="Phase",value="Score",CT,ET,RT) %>% convert_as_factor(Treatment,Phase)
write.csv(cpp,"lfcpp.csv")
# sample rows
csample<-cpp[sample(c(1:nrow(cpp)),6),]

# check for outliers
(cpp %>% group_by(Phase,Treatment) %>% identify_outliers(Score) -> ols)

# count per group
cnt<-count(group_by(cpp,Treatment))
cnt$n<-cnt$n/3

# summary statistics
cpp %>% group_by(Phase,Treatment) %>% get_summary_stats(Score) -> sumstats

# visualize means
gcpp<-cpp %>% group_by(Treatment,Phase) %>% summarise(mean(Score))
gcpp<-rbind(gcpp[4:9,],gcpp[c(1:3,10:12),])
ggplot(gcpp,aes(x=Phase,y=gcpp$`mean(Score)`,fill=Treatment))+geom_bar(stat="identity",position="dodge")+ggtitle("Morphine CPP, Extinction, and oCRF-Induced Reinstatement")+theme(plot.title = element_text(hjust=0.5))+ylab("Time in drug-paired side minus time in unpaired side (s)")
ggsave("mean_bar.jpeg")

# test for normality
cpp %>% group_by(Phase,Treatment) %>% shapiro_test(Score) -> testshapiro
min(testshapiro$p)
# QQ plot
ggqqplot(cpp,"Score",)+facet_grid(Phase~Treatment)+ggtitle("QQ Plot of score by treatment and phase")+theme(plot.title = element_text(hjust=0.5))
ggsave("qq_plot.jpeg")

# test for homoscedasticity
cpp %>% group_by(Phase) %>% levene_test(Score~Treatment) -> testlevene
cpp %>% group_by(Treatment) %>% levene_test(Score~Phase) -> testlevene2

# test for homogeneity of covariance
box_m(cpp[,"Score",drop=F],cpp$Treatment)

# sphericity checked for automatically in anova_test()

# two-way mixed anova tests
# oCRF
(raovcrf<-anova_test(data=cpp[cpp$Treatment=='1. oCRF'|cpp$Treatment=='2. vehicle',],dv=Score,wid=ID,between=Treatment,within=Phase))
raovcrft<-get_anova_table(raovcrf,correction=c("GG"))
# NBI
(raovnbi<-anova_test(data=cpp[cpp$Treatment=='3. NBI+swim'|cpp$Treatment=='4. vehicle+swim',],dv=Score,wid=ID,between=Treatment,within=Phase))
raovnbi<-get_anova_table(raovnbi,correction=c("GG"))

# # preplanned t-tests
# t.test(cpp[cpp$Treatment=='1. oCRF' & cpp$Phase=='RT',4],cpp[cpp$Treatment=='2. vehicle' & cpp$Phase=='RT',4],var.equal=T)
# t.test(cpp[cpp$Treatment=='3. NBI+swim' & cpp$Phase=='RT',4],cpp[cpp$Treatment=='4. vehicle+swim' & cpp$Phase=='RT',4],var.equal=T)

# post-hoc t-tests
# pairwise comparison between phases
pwcp<-cpp %>% group_by(Treatment) %>% pairwise_t_test(Score~Phase)
# pairwise comparison between treatments
pwct<-cpp %>% group_by(Phase) %>% pairwise_t_test(Score~Treatment)
# FDR correction
allp<-c(pwcp$p,pwct$p)
adj.p<-p.adjust(allp,method="fdr")
adj.p.signif<-matrix(rep("ns",length(adj.p)),nrow=length(adj.p),ncol=1)
for (i in c(1:length(adj.p))){
  if (adj.p[i] < 0.05){
    adj.p.signif[i]<-'*'
  }
  if (adj.p[i] < 0.01){
    adj.p.signif[i]<-'**'
  }
  if (adj.p[i] < 0.001){
    adj.p.signif[i]<-'***'
  }
}
npwc<-data.frame(bind_rows(pwcp[,1:8],pwct[,1:8]))
npwc<-data.frame(npwc[,1],npwc[,9],npwc[,2:8],adj.p,adj.p.signif)
colnames(npwc)<-c("Treatment","Phase",colnames(npwc)[3:11])

# visualize
pwc<-pwct %>% add_xy_position(x="Phase")
ggboxplot(cpp,x="Phase",y="Score",color="Treatment") + stat_pvalue_manual(pwc,tip.length=0,hide.ns=T)+ggtitle("Scores by treatment group and phase")+theme(plot.title = element_text(hjust=0.5))
ggsave("boxplot.jpeg")


# Relative Risk and Odds Ratio of Reinstatement across oCRF and vehicle groups
cpp<-read.csv("2021CPPstudy.csv")
# separate into experimental and control groups
crf<-filter(cpp,Treatment=="oCRF")
crfctrl<-filter(cpp,Treatment=="vehicle")
# relative risk of reinstatement for CRF experiment
PPcrfR<-(sum(crf$RT>=100)/(nrow(crf)))
PPcrfctrlR<-(sum(crfctrl$RT>=100)/(nrow(crfctrl)))
(RRcrfR<-PPcrfR/PPcrfctrlR)
# OR
OcrfR<-PPcrfR/(1-PPcrfR)
OcrfctrlR<-PPcrfctrlR/(1-PPcrfctrlR)
(ORcrfR<-OcrfR/OcrfctrlR)

# kmeans clustering of numerical behavioral data
cpp<-read.csv("2021CPPstudy.csv")
bdcpp<-cpp[,5:7]
trtl<-unlist(unique(cpp$Treatment))
cluster<-function(k){
  km<-kmeans(bdcpp,k)
  kmcpp<-data.frame(cpp,km$cluster)
  kmm<-matrix(rep(0,4*k),nrow=4,ncol=k)
  for (i in c(1:4)){
    for (j in c(1:k)){
      kmm[i,j]<-sum(kmcpp[cpp$Treatment==trtl[i],ncol(kmcpp)]==j)
    }
  }
  cldf<-data.frame(kmm)
  row.names(cldf)<-trtl
  colnames(cldf)<-c(sprintf("cluster %s",1:k))
  print("clusters:")
  print(km$centers)
  print("treatment groups:")
  print(cldf)
  return(cldf)
}
for (k in c(2:5)){cluster(k)}

