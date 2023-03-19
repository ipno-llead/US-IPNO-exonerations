library(fs)
library(tidyverse)

f <- tibble(filename = fs::dir_ls("input/wrongful-convictions-docs", recurse = T, type = "file"))

info <- f %>%
    mutate(extension = tools::file_ext(filename),
           size = fs::file_size(filename))

info %>% group_by(extension) %>%
    summarise(size = sum(size),
              n_file = n()) %>%
    arrange(desc(size)) %>%
    print(n=Inf)

