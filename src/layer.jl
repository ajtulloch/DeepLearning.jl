immutable GradientState
    vol::Vol
    multiplierL1::Float64
    multiplierL2::Float64
end

abstract Layer

outputSize(self::Layer) = size(self.outputActivation.w)
inputSize(self::Layer) = size(self.inputActivation.w)

## +----------+
## |          |
## |          |
## |          |
## |          |
## |          |
## +----------+
## |..........|
## |..........|
## |..........<------------+  Input Activation
## |..........|            |
## |..........|            |
## +----------+            |
##                         |
##                         |
## +----------+            |
## |..........|            |
## |..........|            |
## |..........>------------+  Input Activation
## |..........|
## |..........|
## +----------+
## |          |
## |          |
## |          <------------+  Input Activation
## |          |            |
## |          |            |
## +----------+            |
##                         |
##                         |
##                         |
## +----------+            |
## |          |            |
## |          |            |
## |          >------------+  Output Activation
## |          |
## |          |
## +----------+
## |..........|
## |..........|
## |..........>------------+  Activation from lower levels
## |..........|            |
## |..........|            |
## +----------+            |
