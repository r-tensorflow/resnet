if(getRversion() >= "2.15.1")  utils::globalVariables(c("."))


#' @importFrom keras layer_batch_normalization layer_activation
resnet_layer <- function(inputs, num_filters = 16, kernel_size = 3, strides = 1,
                         activation = "relu", batch_normalization = TRUE,
                         conv_first = TRUE) {

  conv <- keras::layer_conv_2d(
    filters = num_filters,
    kernel_size = kernel_size,
    strides = strides,
    padding = 'same',
    kernel_initializer = 'he_normal',
    kernel_regularizer = keras::regularizer_l2(1e-4)
  )

  inputs %>%
    {if (conv_first) conv(.)} %>%
    {if (batch_normalization) layer_batch_normalization(.) else .} %>%
    {if (!is.null(activation)) layer_activation(., activation = activation) else .} %>%
    {if (!conv_first) conv(.) else .}
}

#' ResNet Version 1 Model
#'
#' Stacks of 2 x (3 x 3) Conv2D-BN-ReLU
#' Last ReLU is after the shortcut connection.
#' At the beginning of each stage, the feature map size is halved (downsampled)
#' by a convolutional layer with strides=2, while the number of filters is
#' doubled. Within each stage, the layers have the same number filters and the
#' same number of filters.
#' Features maps sizes:
#' stage 0: 32x32, 16
#' stage 1: 16x16, 32
#' stage 2:  8x8,  64
#' The Number of parameters is approx the same as Table 6 of the ResNet paper:
#' ResNet20 0.27M
#' ResNet32 0.46M
#' ResNet44 0.66M
#' ResNet56 0.85M
#' ResNet110 1.7M
#'
#' @references
#' Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian
#' Sun. Deep residual learning for image recognition.
#'
#' @param input_shape shape of input image tensor
#' @param depth number of core convolutional layers
#' @param num_filters initial number of filters
#' @param num_classes number of classes
#'
#' @export
resnet_v1 <- function(input_shape, depth, num_filters = 16, num_classes=10) {

  if ((depth - 2) %% 6 != 0)
    stop("depth should be 6n+2")

  # Start model definition.
  num_res_blocks <- as.integer((depth - 2) / 6)

  inputs <- keras::layer_input(shape = input_shape)
  x <- resnet_layer(inputs = inputs)

  for (stack in seq_len(3)) {
    for (res_block in seq_len(num_res_blocks)) {
      if (stack > 1 && res_block == 1) { # first layer but not first stack
        strides <- 2
      } else {
        strides <- 1
      }

      y <- x %>%
        resnet_layer(num_filters = num_filters, strides = strides) %>%
        resnet_layer(num_filters = num_filters, strides = strides, activation = NULL)

      if (stack > 0 && res_block == 0) {
        # linear projection residual shortcut connection to match
        # changed dims
        x <- x %>% resnet_layer(
          num_filters=num_filters,
          kernel_size = 1,
          strides = strides,
          activation = NULL,
          batch_normalization = FALSE
        )
        x <- keras::layer_add(list(x, y)) %>%
          keras::layer_activation("relu")
      }
    }
    num_filters = 2*num_filters
  }

  outputs <- x %>%
    keras::layer_average_pooling_2d(pool_size = c(8,8)) %>%
    keras::layer_flatten() %>%
    keras::layer_dense(
      units = num_classes,
      activation = "softmax",
      kernel_initializer = "he_normal"
    )

  keras::keras_model(inputs, outputs)
}

#' ResNet Version 2 Model
#'
#' Stacks of (1 x 1)-(3 x 3)-(1 x 1) BN-ReLU-Conv2D or also known as
#' bottleneck layer
#' First shortcut connection per layer is 1 x 1 Conv2D.
#' Second and onwards shortcut connection is identity.
#' At the beginning of each stage, the feature map size is halved (downsampled)
#' by a convolutional layer with strides=2, while the number of filter maps is
#' doubled. Within each stage, the layers have the same number filters and the
#' same filter map sizes.
#' Features maps sizes:
#'   conv1  : 32x32,  16
#' stage 0: 32x32,  64
#' stage 1: 16x16, 128
#' stage 2:  8x8,  256
#'
#' @references
#' He, Kaiming & Zhang, Xiangyu & Ren, Shaoqing & Sun, Jian. (2016). Identity Mappings
#' in Deep Residual Networks. 9908. 630-645. 10.1007/978-3-319-46493-0_38.
#'
#' @inheritParams resnet_v1
#'
#' @export
resnet_v2 <- function(input_shape, depth, num_filters = 16, num_classes = 10) {

  if ((depth - 2) %% 9 != 0)
    stop("depth should be 9n+2 (eg 56 or 110 in [b])")

  num_res_blocks <- as.integer((depth - 2) / 9)

  inputs <- keras::layer_input(shape = input_shape)
  x <- resnet_layer(inputs, num_filters = num_filters, conv_first = TRUE)

  for (stage in seq_len(3)) {
    for (res_block in seq_len(num_res_blocks)) {

      activation <- "relu"
      batch_normalization <- TRUE
      strides <- 1

      if (stage == 1) {
        num_filters_out <- num_filters * 4
        if (res_block == 1) {
          activation <- NULL
          batch_normalization <- FALSE
        }
      } else {
        num_filters_out <- num_filters * 2
        if (res_block == 1) {
          strides <- 2
        }
      }

      y <- x %>%
        resnet_layer(
          num_filters = num_filters, kernel_size=1,
          strides = strides,
          activation = activation,
          batch_normalization = batch_normalization,
          conv_first = FALSE
        ) %>%
        resnet_layer(
          num_filters = num_filters,
          conv_first = FALSE
        ) %>%
        resnet_layer(
          num_filters = num_filters_out,
          kernel_size = 1,
          conv_first = FALSE
        )

      if (res_block == 1) {
        x <- x %>%
          resnet_layer(
            num_filters = num_filters_out,
            kernel_size = 1,
            strides = strides,
            activation = NULL,
            batch_normalization = FALSE
          )
      }


      x <- keras::layer_add(list(x, y))

    }
    num_filters <- num_filters_out
  }

  outputs <- x %>%
    keras::layer_batch_normalization() %>%
    keras::layer_activation("relu") %>%
    keras::layer_average_pooling_2d(pool_size = 8) %>%
    keras::layer_flatten() %>%
    keras::layer_dense(
      units = num_classes,
      activation = "softmax",
      kernel_initializer = "he_normal"
    )

  keras::keras_model(inputs, outputs)
}
