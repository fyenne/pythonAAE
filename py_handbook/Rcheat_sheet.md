# hypothesis testing

```R
Library(car)

LinearHypothesis(model, "variable1 = variable2", test=c("Chisq", "F")) 
lfe::waldtest(model, ~ variable1-variable2, type = "cluster")
#--------------------------------------------
linearHypothesis(fitQ6_c, c("monthf9 = monthf10","monthf9 = monthf11"), test = c("F" ))
lfe::waldtest(fitQ6_c, ~ monthf9-monthf10|monthf9-monthf11, type = "cluster")
```

# time machine

```R
library(lubricate)
wday(`date`) # day of the week.

covid = covid %>% 
  mutate(date = ymd(date)) %>% 
  mutate_at(vars(date), funs(year, 
  month, day))

```

# pinyin

``` Pinyin 
py(sep = "", dic = pydic(method = 'toneless', dic = "pinyin2"))
```

# stargazer

stargazer save to file.
```{r}
mod_stargazer <- function(output.file, ...) {
  output <- capture.output(stargazer(...))
  cat(paste(output, collapse = "\n"), "\n", file=output.file, append=F)
}
```

```{r}
mod_stargazer("reg_air.txt", reg_air, type = "text")
```

