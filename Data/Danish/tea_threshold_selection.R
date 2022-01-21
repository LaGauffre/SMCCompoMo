library(SMPracticals)
setwd(dirname(rstudioapi::getSourceEditorContext()$path))
data(danish)
summary(danish)
thresholds = c(dAMSE(danish)$threshold, DK(danish)$threshold, eye(danish)$threshold, 
  ggplot(danish, nexceed = min(danish) - 1)$threshold, 
  GH(danish)$threshold, gomes(danish)$threshold,
  hall(danish)$threshold, Himp(danish)$threshold)
methods = c("AMSE Caeiro and Gomes", "Drees and Kauffman", 
            "eye Danielsson et al.", 
            "Gerstengarbe plot Gerstengarbe and Werner", 
            "exp test Guillou and Hall", 
            "double bootsrap Gomes et al.",
            "Single bootstrap Hall",
            "Hall and Welsh")
thresholds_df = data.frame(threshold = thresholds, 
                           method = methods)

write.csv(thresholds_df, "tea_threshold.csv")
