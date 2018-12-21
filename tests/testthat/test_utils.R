context("utils")
set.seed(1)
a <- matrix(rnorm(10))
A <- matrix(rnorm(50), 5, 10)
B <- tcrossprod(A, A)
C <- matrix(rnorm(100), 10, 10)
D <- t(A)
e <- matrix(rnorm(5))

a_tf <- tf$constant(a, name = "a", dtype = "float32")
A_tf <- tf$constant(A, name = "A", dtype = "float32")
B_tf <- tf$constant(B, name = "B", dtype = "float32")
C_tf <- tf$constant(C, name = "C", dtype = "float32")
D_tf <- tf$constant(D, name = "D", dtype = "float32")
e_tf <- tf$constant(e, dtype = "float32")
cholB_tf <- tf$cholesky_upper(B_tf)

test_that("logdet works as expected", {
  logdetB_raw <- as.numeric(determinant(B)$modulus)
  logdetB <- logdet(chol(B))
  expect_equal(logdetB_raw, logdetB)

  ## Tensorflow
  logdetB_tf_comp <- logdet_tf(cholB_tf) %>% tf$Session()$run()
  expect_equal(logdetB_raw, logdetB_tf_comp, tolerance = 1e-6)
})

test_that("safe_chol works", {
  R0 <- chol(B + diag(nrow(B)) * 1e-6)
  R1 <- safe_chol(B)
  R2 <- tf$Session()$run(safe_chol_tf(B_tf))
  expect_equal(R0, R1)
  expect_equal(R1, R2, tolerance = 1e-6)
})

test_that("atBa works as expected", {
  atBa_raw <- t(a) %*% C %*% a
  expect_equal(atBa_raw, atBa(a, C))

  ## Tensorflow
  atBa_tf_comp <-
    atBa_tf(a_tf, C_tf) %>%
    tf$Session()$run()
  expect_equal(atBa_raw, atBa_tf_comp, tolerance = 1e-6)

})

test_that("ABinvAt works as expected", {
  ABinvAt_raw <- D %*% solve(B) %*% t(D)
  expect_equal(ABinvAt_raw, ABinvAt(D, chol(B)))

 ## Tensorflow
 ABinvAt_tf_comp <-
    ABinvAt_tf(D_tf, cholB_tf) %>%
    tf$Session()$run()
  expect_equal(ABinvAt_raw, ABinvAt_tf_comp, tolerance = 1e-6)

})

test_that("AtBA_p_C works as expected", {
  AtBA_p_C_raw <- t(A) %*% B %*% A + C
  expect_equal(AtBA_p_C_raw, AtBA_p_C(A, chol(B), C))

  ## Tensorflow
  AtBA_p_C_tf_comp <-
    AtBA_p_C_tf(A_tf, cholB_tf, C_tf) %>%
    tf$Session()$run()
  expect_equal(AtBA_p_C_raw, AtBA_p_C_tf_comp, tolerance = 1e-6)
})

test_that("entropy works as expected", {
  set.seed(1)
  s <- matrix(runif(10), 5, 2)
  S1 <- diag(s[,1])
  S2 <- diag(s[,2])
  entropy_raw <- 0.5 * (log(det(S1)) + log(det(S2)))
  expect_equal(entropy_raw, entropy(s))

  ## Tensorflow
  s_tf <- tf$constant(s, name = "s", dtype = "float32")
  entropy_tf_comp <-  entropy_tf(s_tf) %>% tf$Session()$run()
  expect_equal(entropy_raw, entropy_tf_comp, tolerance = 1e-6)
})

test_that("chol2inv works as expected", {
  Binv_raw <- solve(B)
  expect_equal(Binv_raw, chol2inv(chol(B)))

  ## Tensorflow
  Binv_tf_comp <-  chol2inv_tf(R = cholB_tf) %>% tf$Session()$run()
  expect_equal(Binv_raw, Binv_tf_comp, tolerance = 1e-6)
})


test_that("get_depvars works", {
  f1 <- y ~ x
  f2 <- ~ x
  f3 <- cbind(y1, y2) ~ x
  expect_equal(get_depvars(f1), "y")
  expect_equal(get_depvars(f2), NULL)
  expect_equal(get_depvars(f3), c("y1", "y2"))
})

test_that("list_to_listtf works", {
  l <- list(matrix(1:6, 3, 2),
            matrix(1:10,2, 5))
  l_tf <- list_to_listtf(l, name = "mylist")
  sess_run <- tf$Session()$run
  expect_equal(l[[1]], sess_run(l_tf[[1]]))
  expect_equal(l[[2]], sess_run(l_tf[[2]]))
})

test_that("pinvsolve works", {
  expect_equal(pinvsolve(B, e), solve(B, e))
  expect_equal(pinvsolve_tf(B_tf, e_tf) %>% tf$Session()$run(), solve(B, e),
               tolerance = 1e-5)
})

test_that("proc.m_inducing works", {
  expect_equal(proc_m.inducing(5L, 2), c(5L, 5L))
  expect_equal(proc_m.inducing(5L, 1), 5L)
})
