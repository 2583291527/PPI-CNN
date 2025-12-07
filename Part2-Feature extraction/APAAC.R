library(openxlsx)
library(ftrCOOL)

fileExcel <- 
sequences_data <- read.xlsx(fileExcel,colNames = FALSE)

sequences <- sequences_data[[1]]

# 运行 APAAC 函数
mat <- APAAC(seqs = sequences, aaIDX = c("ARGP820101", "HOPT810101"), lambda = 30, 
             w = 0.5, l = 1, threshold = 1, label = c())

# 将结果保存到Excel文件
write.xlsx(mat, file = ,colNames = FALSE)