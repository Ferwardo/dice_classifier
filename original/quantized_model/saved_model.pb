½ß4
£ó
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype

Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

s
FakeQuantWithMinMaxVars

inputs
min
max
outputs"
num_bitsint"
narrow_rangebool( 
}
!FakeQuantWithMinMaxVarsPerChannel

inputs
min
max
outputs"
num_bitsint"
narrow_rangebool( 
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	

MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C

MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( 

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
Á
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ¨
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.10.02v2.10.0-rc3-6-g359c3cdfc5f8ø.
~
Adam/dense_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_2/bias/v
w
'Adam/dense_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_2/bias/v*
_output_shapes
:*
dtype0

Adam/dense_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*&
shared_nameAdam/dense_2/kernel/v

)Adam/dense_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_2/kernel/v*
_output_shapes
:	*
dtype0

Adam/dense_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_1/bias/v
x
'Adam/dense_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/v*
_output_shapes	
:*
dtype0

Adam/dense_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*&
shared_nameAdam/dense_1/kernel/v

)Adam/dense_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/v* 
_output_shapes
:
*
dtype0
{
Adam/dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/dense/bias/v
t
%Adam/dense/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense/bias/v*
_output_shapes	
:*
dtype0

Adam/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*$
shared_nameAdam/dense/kernel/v
}
'Adam/dense/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/v* 
_output_shapes
:
*
dtype0

Adam/conv2d_4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/conv2d_4/bias/v
y
(Adam/conv2d_4/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_4/bias/v*
_output_shapes
: *
dtype0

Adam/conv2d_4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@ *'
shared_nameAdam/conv2d_4/kernel/v

*Adam/conv2d_4/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_4/kernel/v*&
_output_shapes
:@ *
dtype0

Adam/conv2d_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/conv2d_3/bias/v
y
(Adam/conv2d_3/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_3/bias/v*
_output_shapes
:@*
dtype0

Adam/conv2d_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*'
shared_nameAdam/conv2d_3/kernel/v

*Adam/conv2d_3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_3/kernel/v*&
_output_shapes
:@@*
dtype0

Adam/conv2d_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/conv2d_2/bias/v
y
(Adam/conv2d_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_2/bias/v*
_output_shapes
:@*
dtype0

Adam/conv2d_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*'
shared_nameAdam/conv2d_2/kernel/v

*Adam/conv2d_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_2/kernel/v*&
_output_shapes
: @*
dtype0

Adam/conv2d_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/conv2d_1/bias/v
y
(Adam/conv2d_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/bias/v*
_output_shapes
: *
dtype0

Adam/conv2d_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *'
shared_nameAdam/conv2d_1/kernel/v

*Adam/conv2d_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/kernel/v*&
_output_shapes
:  *
dtype0
|
Adam/conv2d/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/conv2d/bias/v
u
&Adam/conv2d/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d/bias/v*
_output_shapes
: *
dtype0

Adam/conv2d/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/conv2d/kernel/v

(Adam/conv2d/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d/kernel/v*&
_output_shapes
: *
dtype0
~
Adam/dense_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_2/bias/m
w
'Adam/dense_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_2/bias/m*
_output_shapes
:*
dtype0

Adam/dense_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*&
shared_nameAdam/dense_2/kernel/m

)Adam/dense_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_2/kernel/m*
_output_shapes
:	*
dtype0

Adam/dense_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_1/bias/m
x
'Adam/dense_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/m*
_output_shapes	
:*
dtype0

Adam/dense_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*&
shared_nameAdam/dense_1/kernel/m

)Adam/dense_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/m* 
_output_shapes
:
*
dtype0
{
Adam/dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/dense/bias/m
t
%Adam/dense/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense/bias/m*
_output_shapes	
:*
dtype0

Adam/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*$
shared_nameAdam/dense/kernel/m
}
'Adam/dense/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/m* 
_output_shapes
:
*
dtype0

Adam/conv2d_4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/conv2d_4/bias/m
y
(Adam/conv2d_4/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_4/bias/m*
_output_shapes
: *
dtype0

Adam/conv2d_4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@ *'
shared_nameAdam/conv2d_4/kernel/m

*Adam/conv2d_4/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_4/kernel/m*&
_output_shapes
:@ *
dtype0

Adam/conv2d_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/conv2d_3/bias/m
y
(Adam/conv2d_3/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_3/bias/m*
_output_shapes
:@*
dtype0

Adam/conv2d_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*'
shared_nameAdam/conv2d_3/kernel/m

*Adam/conv2d_3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_3/kernel/m*&
_output_shapes
:@@*
dtype0

Adam/conv2d_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/conv2d_2/bias/m
y
(Adam/conv2d_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_2/bias/m*
_output_shapes
:@*
dtype0

Adam/conv2d_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*'
shared_nameAdam/conv2d_2/kernel/m

*Adam/conv2d_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_2/kernel/m*&
_output_shapes
: @*
dtype0

Adam/conv2d_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/conv2d_1/bias/m
y
(Adam/conv2d_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/bias/m*
_output_shapes
: *
dtype0

Adam/conv2d_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *'
shared_nameAdam/conv2d_1/kernel/m

*Adam/conv2d_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/kernel/m*&
_output_shapes
:  *
dtype0
|
Adam/conv2d/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/conv2d/bias/m
u
&Adam/conv2d/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d/bias/m*
_output_shapes
: *
dtype0

Adam/conv2d/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/conv2d/kernel/m

(Adam/conv2d/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d/kernel/m*&
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
p
dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_2/bias
i
 dense_2/bias/Read/ReadVariableOpReadVariableOpdense_2/bias*
_output_shapes
:*
dtype0
y
dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*
shared_namedense_2/kernel
r
"dense_2/kernel/Read/ReadVariableOpReadVariableOpdense_2/kernel*
_output_shapes
:	*
dtype0
q
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_1/bias
j
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes	
:*
dtype0
z
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_namedense_1/kernel
s
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel* 
_output_shapes
:
*
dtype0
m

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
dense/bias
f
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes	
:*
dtype0
v
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_namedense/kernel
o
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel* 
_output_shapes
:
*
dtype0
r
conv2d_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_4/bias
k
!conv2d_4/bias/Read/ReadVariableOpReadVariableOpconv2d_4/bias*
_output_shapes
: *
dtype0

conv2d_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@ * 
shared_nameconv2d_4/kernel
{
#conv2d_4/kernel/Read/ReadVariableOpReadVariableOpconv2d_4/kernel*&
_output_shapes
:@ *
dtype0
r
conv2d_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_3/bias
k
!conv2d_3/bias/Read/ReadVariableOpReadVariableOpconv2d_3/bias*
_output_shapes
:@*
dtype0

conv2d_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@* 
shared_nameconv2d_3/kernel
{
#conv2d_3/kernel/Read/ReadVariableOpReadVariableOpconv2d_3/kernel*&
_output_shapes
:@@*
dtype0
r
conv2d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_2/bias
k
!conv2d_2/bias/Read/ReadVariableOpReadVariableOpconv2d_2/bias*
_output_shapes
:@*
dtype0

conv2d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @* 
shared_nameconv2d_2/kernel
{
#conv2d_2/kernel/Read/ReadVariableOpReadVariableOpconv2d_2/kernel*&
_output_shapes
: @*
dtype0
r
conv2d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_1/bias
k
!conv2d_1/bias/Read/ReadVariableOpReadVariableOpconv2d_1/bias*
_output_shapes
: *
dtype0

conv2d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  * 
shared_nameconv2d_1/kernel
{
#conv2d_1/kernel/Read/ReadVariableOpReadVariableOpconv2d_1/kernel*&
_output_shapes
:  *
dtype0
n
conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d/bias
g
conv2d/bias/Read/ReadVariableOpReadVariableOpconv2d/bias*
_output_shapes
: *
dtype0
~
conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d/kernel
w
!conv2d/kernel/Read/ReadVariableOpReadVariableOpconv2d/kernel*&
_output_shapes
: *
dtype0

 quant_dense_2/pre_activation_maxVarHandleOp*
_output_shapes
: *
dtype0*
shape: *1
shared_name" quant_dense_2/pre_activation_max

4quant_dense_2/pre_activation_max/Read/ReadVariableOpReadVariableOp quant_dense_2/pre_activation_max*
_output_shapes
: *
dtype0

 quant_dense_2/pre_activation_minVarHandleOp*
_output_shapes
: *
dtype0*
shape: *1
shared_name" quant_dense_2/pre_activation_min

4quant_dense_2/pre_activation_min/Read/ReadVariableOpReadVariableOp quant_dense_2/pre_activation_min*
_output_shapes
: *
dtype0

quant_dense_2/kernel_maxVarHandleOp*
_output_shapes
: *
dtype0*
shape: *)
shared_namequant_dense_2/kernel_max
}
,quant_dense_2/kernel_max/Read/ReadVariableOpReadVariableOpquant_dense_2/kernel_max*
_output_shapes
: *
dtype0

quant_dense_2/kernel_minVarHandleOp*
_output_shapes
: *
dtype0*
shape: *)
shared_namequant_dense_2/kernel_min
}
,quant_dense_2/kernel_min/Read/ReadVariableOpReadVariableOpquant_dense_2/kernel_min*
_output_shapes
: *
dtype0

quant_dense_2/optimizer_stepVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_namequant_dense_2/optimizer_step

0quant_dense_2/optimizer_step/Read/ReadVariableOpReadVariableOpquant_dense_2/optimizer_step*
_output_shapes
: *
dtype0

quant_dropout_1/optimizer_stepVarHandleOp*
_output_shapes
: *
dtype0*
shape: */
shared_name quant_dropout_1/optimizer_step

2quant_dropout_1/optimizer_step/Read/ReadVariableOpReadVariableOpquant_dropout_1/optimizer_step*
_output_shapes
: *
dtype0

!quant_dense_1/post_activation_maxVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!quant_dense_1/post_activation_max

5quant_dense_1/post_activation_max/Read/ReadVariableOpReadVariableOp!quant_dense_1/post_activation_max*
_output_shapes
: *
dtype0

!quant_dense_1/post_activation_minVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!quant_dense_1/post_activation_min

5quant_dense_1/post_activation_min/Read/ReadVariableOpReadVariableOp!quant_dense_1/post_activation_min*
_output_shapes
: *
dtype0

quant_dense_1/kernel_maxVarHandleOp*
_output_shapes
: *
dtype0*
shape: *)
shared_namequant_dense_1/kernel_max
}
,quant_dense_1/kernel_max/Read/ReadVariableOpReadVariableOpquant_dense_1/kernel_max*
_output_shapes
: *
dtype0

quant_dense_1/kernel_minVarHandleOp*
_output_shapes
: *
dtype0*
shape: *)
shared_namequant_dense_1/kernel_min
}
,quant_dense_1/kernel_min/Read/ReadVariableOpReadVariableOpquant_dense_1/kernel_min*
_output_shapes
: *
dtype0

quant_dense_1/optimizer_stepVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_namequant_dense_1/optimizer_step

0quant_dense_1/optimizer_step/Read/ReadVariableOpReadVariableOpquant_dense_1/optimizer_step*
_output_shapes
: *
dtype0

quant_dropout/optimizer_stepVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_namequant_dropout/optimizer_step

0quant_dropout/optimizer_step/Read/ReadVariableOpReadVariableOpquant_dropout/optimizer_step*
_output_shapes
: *
dtype0

quant_dense/post_activation_maxVarHandleOp*
_output_shapes
: *
dtype0*
shape: *0
shared_name!quant_dense/post_activation_max

3quant_dense/post_activation_max/Read/ReadVariableOpReadVariableOpquant_dense/post_activation_max*
_output_shapes
: *
dtype0

quant_dense/post_activation_minVarHandleOp*
_output_shapes
: *
dtype0*
shape: *0
shared_name!quant_dense/post_activation_min

3quant_dense/post_activation_min/Read/ReadVariableOpReadVariableOpquant_dense/post_activation_min*
_output_shapes
: *
dtype0

quant_dense/kernel_maxVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_namequant_dense/kernel_max
y
*quant_dense/kernel_max/Read/ReadVariableOpReadVariableOpquant_dense/kernel_max*
_output_shapes
: *
dtype0

quant_dense/kernel_minVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_namequant_dense/kernel_min
y
*quant_dense/kernel_min/Read/ReadVariableOpReadVariableOpquant_dense/kernel_min*
_output_shapes
: *
dtype0

quant_dense/optimizer_stepVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_namequant_dense/optimizer_step

.quant_dense/optimizer_step/Read/ReadVariableOpReadVariableOpquant_dense/optimizer_step*
_output_shapes
: *
dtype0

quant_flatten/optimizer_stepVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_namequant_flatten/optimizer_step

0quant_flatten/optimizer_step/Read/ReadVariableOpReadVariableOpquant_flatten/optimizer_step*
_output_shapes
: *
dtype0

$quant_max_pooling2d_2/optimizer_stepVarHandleOp*
_output_shapes
: *
dtype0*
shape: *5
shared_name&$quant_max_pooling2d_2/optimizer_step

8quant_max_pooling2d_2/optimizer_step/Read/ReadVariableOpReadVariableOp$quant_max_pooling2d_2/optimizer_step*
_output_shapes
: *
dtype0

"quant_conv2d_4/post_activation_maxVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"quant_conv2d_4/post_activation_max

6quant_conv2d_4/post_activation_max/Read/ReadVariableOpReadVariableOp"quant_conv2d_4/post_activation_max*
_output_shapes
: *
dtype0

"quant_conv2d_4/post_activation_minVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"quant_conv2d_4/post_activation_min

6quant_conv2d_4/post_activation_min/Read/ReadVariableOpReadVariableOp"quant_conv2d_4/post_activation_min*
_output_shapes
: *
dtype0

quant_conv2d_4/kernel_maxVarHandleOp*
_output_shapes
: *
dtype0*
shape: **
shared_namequant_conv2d_4/kernel_max

-quant_conv2d_4/kernel_max/Read/ReadVariableOpReadVariableOpquant_conv2d_4/kernel_max*
_output_shapes
: *
dtype0

quant_conv2d_4/kernel_minVarHandleOp*
_output_shapes
: *
dtype0*
shape: **
shared_namequant_conv2d_4/kernel_min

-quant_conv2d_4/kernel_min/Read/ReadVariableOpReadVariableOpquant_conv2d_4/kernel_min*
_output_shapes
: *
dtype0

quant_conv2d_4/optimizer_stepVarHandleOp*
_output_shapes
: *
dtype0*
shape: *.
shared_namequant_conv2d_4/optimizer_step

1quant_conv2d_4/optimizer_step/Read/ReadVariableOpReadVariableOpquant_conv2d_4/optimizer_step*
_output_shapes
: *
dtype0

"quant_conv2d_3/post_activation_maxVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"quant_conv2d_3/post_activation_max

6quant_conv2d_3/post_activation_max/Read/ReadVariableOpReadVariableOp"quant_conv2d_3/post_activation_max*
_output_shapes
: *
dtype0

"quant_conv2d_3/post_activation_minVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"quant_conv2d_3/post_activation_min

6quant_conv2d_3/post_activation_min/Read/ReadVariableOpReadVariableOp"quant_conv2d_3/post_activation_min*
_output_shapes
: *
dtype0

quant_conv2d_3/kernel_maxVarHandleOp*
_output_shapes
: *
dtype0*
shape:@**
shared_namequant_conv2d_3/kernel_max

-quant_conv2d_3/kernel_max/Read/ReadVariableOpReadVariableOpquant_conv2d_3/kernel_max*
_output_shapes
:@*
dtype0

quant_conv2d_3/kernel_minVarHandleOp*
_output_shapes
: *
dtype0*
shape:@**
shared_namequant_conv2d_3/kernel_min

-quant_conv2d_3/kernel_min/Read/ReadVariableOpReadVariableOpquant_conv2d_3/kernel_min*
_output_shapes
:@*
dtype0

quant_conv2d_3/optimizer_stepVarHandleOp*
_output_shapes
: *
dtype0*
shape: *.
shared_namequant_conv2d_3/optimizer_step

1quant_conv2d_3/optimizer_step/Read/ReadVariableOpReadVariableOpquant_conv2d_3/optimizer_step*
_output_shapes
: *
dtype0

"quant_conv2d_2/post_activation_maxVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"quant_conv2d_2/post_activation_max

6quant_conv2d_2/post_activation_max/Read/ReadVariableOpReadVariableOp"quant_conv2d_2/post_activation_max*
_output_shapes
: *
dtype0

"quant_conv2d_2/post_activation_minVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"quant_conv2d_2/post_activation_min

6quant_conv2d_2/post_activation_min/Read/ReadVariableOpReadVariableOp"quant_conv2d_2/post_activation_min*
_output_shapes
: *
dtype0

quant_conv2d_2/kernel_maxVarHandleOp*
_output_shapes
: *
dtype0*
shape:@**
shared_namequant_conv2d_2/kernel_max

-quant_conv2d_2/kernel_max/Read/ReadVariableOpReadVariableOpquant_conv2d_2/kernel_max*
_output_shapes
:@*
dtype0

quant_conv2d_2/kernel_minVarHandleOp*
_output_shapes
: *
dtype0*
shape:@**
shared_namequant_conv2d_2/kernel_min

-quant_conv2d_2/kernel_min/Read/ReadVariableOpReadVariableOpquant_conv2d_2/kernel_min*
_output_shapes
:@*
dtype0

quant_conv2d_2/optimizer_stepVarHandleOp*
_output_shapes
: *
dtype0*
shape: *.
shared_namequant_conv2d_2/optimizer_step

1quant_conv2d_2/optimizer_step/Read/ReadVariableOpReadVariableOpquant_conv2d_2/optimizer_step*
_output_shapes
: *
dtype0

$quant_max_pooling2d_1/optimizer_stepVarHandleOp*
_output_shapes
: *
dtype0*
shape: *5
shared_name&$quant_max_pooling2d_1/optimizer_step

8quant_max_pooling2d_1/optimizer_step/Read/ReadVariableOpReadVariableOp$quant_max_pooling2d_1/optimizer_step*
_output_shapes
: *
dtype0

"quant_conv2d_1/post_activation_maxVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"quant_conv2d_1/post_activation_max

6quant_conv2d_1/post_activation_max/Read/ReadVariableOpReadVariableOp"quant_conv2d_1/post_activation_max*
_output_shapes
: *
dtype0

"quant_conv2d_1/post_activation_minVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"quant_conv2d_1/post_activation_min

6quant_conv2d_1/post_activation_min/Read/ReadVariableOpReadVariableOp"quant_conv2d_1/post_activation_min*
_output_shapes
: *
dtype0

quant_conv2d_1/kernel_maxVarHandleOp*
_output_shapes
: *
dtype0*
shape: **
shared_namequant_conv2d_1/kernel_max

-quant_conv2d_1/kernel_max/Read/ReadVariableOpReadVariableOpquant_conv2d_1/kernel_max*
_output_shapes
: *
dtype0

quant_conv2d_1/kernel_minVarHandleOp*
_output_shapes
: *
dtype0*
shape: **
shared_namequant_conv2d_1/kernel_min

-quant_conv2d_1/kernel_min/Read/ReadVariableOpReadVariableOpquant_conv2d_1/kernel_min*
_output_shapes
: *
dtype0

quant_conv2d_1/optimizer_stepVarHandleOp*
_output_shapes
: *
dtype0*
shape: *.
shared_namequant_conv2d_1/optimizer_step

1quant_conv2d_1/optimizer_step/Read/ReadVariableOpReadVariableOpquant_conv2d_1/optimizer_step*
_output_shapes
: *
dtype0

"quant_max_pooling2d/optimizer_stepVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"quant_max_pooling2d/optimizer_step

6quant_max_pooling2d/optimizer_step/Read/ReadVariableOpReadVariableOp"quant_max_pooling2d/optimizer_step*
_output_shapes
: *
dtype0

 quant_conv2d/post_activation_maxVarHandleOp*
_output_shapes
: *
dtype0*
shape: *1
shared_name" quant_conv2d/post_activation_max

4quant_conv2d/post_activation_max/Read/ReadVariableOpReadVariableOp quant_conv2d/post_activation_max*
_output_shapes
: *
dtype0

 quant_conv2d/post_activation_minVarHandleOp*
_output_shapes
: *
dtype0*
shape: *1
shared_name" quant_conv2d/post_activation_min

4quant_conv2d/post_activation_min/Read/ReadVariableOpReadVariableOp quant_conv2d/post_activation_min*
_output_shapes
: *
dtype0

quant_conv2d/kernel_maxVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_namequant_conv2d/kernel_max

+quant_conv2d/kernel_max/Read/ReadVariableOpReadVariableOpquant_conv2d/kernel_max*
_output_shapes
: *
dtype0

quant_conv2d/kernel_minVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_namequant_conv2d/kernel_min

+quant_conv2d/kernel_min/Read/ReadVariableOpReadVariableOpquant_conv2d/kernel_min*
_output_shapes
: *
dtype0

quant_conv2d/optimizer_stepVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namequant_conv2d/optimizer_step

/quant_conv2d/optimizer_step/Read/ReadVariableOpReadVariableOpquant_conv2d/optimizer_step*
_output_shapes
: *
dtype0

quantize_layer/optimizer_stepVarHandleOp*
_output_shapes
: *
dtype0*
shape: *.
shared_namequantize_layer/optimizer_step

1quantize_layer/optimizer_step/Read/ReadVariableOpReadVariableOpquantize_layer/optimizer_step*
_output_shapes
: *
dtype0

!quantize_layer/quantize_layer_maxVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!quantize_layer/quantize_layer_max

5quantize_layer/quantize_layer_max/Read/ReadVariableOpReadVariableOp!quantize_layer/quantize_layer_max*
_output_shapes
: *
dtype0

!quantize_layer/quantize_layer_minVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!quantize_layer/quantize_layer_min

5quantize_layer/quantize_layer_min/Read/ReadVariableOpReadVariableOp!quantize_layer/quantize_layer_min*
_output_shapes
: *
dtype0

serving_default_conv2d_inputPlaceholder*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿææ*
dtype0*&
shape:ÿÿÿÿÿÿÿÿÿææ
³
StatefulPartitionedCallStatefulPartitionedCallserving_default_conv2d_input!quantize_layer/quantize_layer_min!quantize_layer/quantize_layer_maxconv2d/kernelquant_conv2d/kernel_minquant_conv2d/kernel_maxconv2d/bias quant_conv2d/post_activation_min quant_conv2d/post_activation_maxconv2d_1/kernelquant_conv2d_1/kernel_minquant_conv2d_1/kernel_maxconv2d_1/bias"quant_conv2d_1/post_activation_min"quant_conv2d_1/post_activation_maxconv2d_2/kernelquant_conv2d_2/kernel_minquant_conv2d_2/kernel_maxconv2d_2/bias"quant_conv2d_2/post_activation_min"quant_conv2d_2/post_activation_maxconv2d_3/kernelquant_conv2d_3/kernel_minquant_conv2d_3/kernel_maxconv2d_3/bias"quant_conv2d_3/post_activation_min"quant_conv2d_3/post_activation_maxconv2d_4/kernelquant_conv2d_4/kernel_minquant_conv2d_4/kernel_maxconv2d_4/bias"quant_conv2d_4/post_activation_min"quant_conv2d_4/post_activation_maxdense/kernelquant_dense/kernel_minquant_dense/kernel_max
dense/biasquant_dense/post_activation_minquant_dense/post_activation_maxdense_1/kernelquant_dense_1/kernel_minquant_dense_1/kernel_maxdense_1/bias!quant_dense_1/post_activation_min!quant_dense_1/post_activation_maxdense_2/kernelquant_dense_2/kernel_minquant_dense_2/kernel_maxdense_2/bias quant_dense_2/pre_activation_min quant_dense_2/pre_activation_max*>
Tin7
523*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*T
_read_only_resource_inputs6
42	
 !"#$%&'()*+,-./012*;
config_proto+)

CPU

GPU2*0J
	
   E 8 *-
f(R&
$__inference_signature_wrapper_145187

NoOpNoOp

ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*Ó
valueÈBÄ B¼

layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
layer_with_weights-4
layer-4
layer_with_weights-5
layer-5
layer_with_weights-6
layer-6
layer_with_weights-7
layer-7
	layer_with_weights-8
	layer-8

layer_with_weights-9

layer-9
layer_with_weights-10
layer-10
layer_with_weights-11
layer-11
layer_with_weights-12
layer-12
layer_with_weights-13
layer-13
layer_with_weights-14
layer-14
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures*
è
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
quantize_layer_min
 quantize_layer_max
!quantizer_vars
"optimizer_step*
Æ
#	variables
$trainable_variables
%regularization_losses
&	keras_api
'__call__
*(&call_and_return_all_conditional_losses
	)layer
*optimizer_step
+_weight_vars
,
kernel_min
-
kernel_max
._quantize_activations
/post_activation_min
0post_activation_max
1_output_quantizers*
ô
2	variables
3trainable_variables
4regularization_losses
5	keras_api
6__call__
*7&call_and_return_all_conditional_losses
	8layer
9optimizer_step
:_weight_vars
;_quantize_activations
<_output_quantizers*
Æ
=	variables
>trainable_variables
?regularization_losses
@	keras_api
A__call__
*B&call_and_return_all_conditional_losses
	Clayer
Doptimizer_step
E_weight_vars
F
kernel_min
G
kernel_max
H_quantize_activations
Ipost_activation_min
Jpost_activation_max
K_output_quantizers*
ô
L	variables
Mtrainable_variables
Nregularization_losses
O	keras_api
P__call__
*Q&call_and_return_all_conditional_losses
	Rlayer
Soptimizer_step
T_weight_vars
U_quantize_activations
V_output_quantizers*
Æ
W	variables
Xtrainable_variables
Yregularization_losses
Z	keras_api
[__call__
*\&call_and_return_all_conditional_losses
	]layer
^optimizer_step
__weight_vars
`
kernel_min
a
kernel_max
b_quantize_activations
cpost_activation_min
dpost_activation_max
e_output_quantizers*
Æ
f	variables
gtrainable_variables
hregularization_losses
i	keras_api
j__call__
*k&call_and_return_all_conditional_losses
	llayer
moptimizer_step
n_weight_vars
o
kernel_min
p
kernel_max
q_quantize_activations
rpost_activation_min
spost_activation_max
t_output_quantizers*
Ê
u	variables
vtrainable_variables
wregularization_losses
x	keras_api
y__call__
*z&call_and_return_all_conditional_losses
	{layer
|optimizer_step
}_weight_vars
~
kernel_min

kernel_max
_quantize_activations
post_activation_min
post_activation_max
_output_quantizers*
ÿ
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses

layer
optimizer_step
_weight_vars
_quantize_activations
_output_quantizers*
ÿ
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses

layer
optimizer_step
_weight_vars
_quantize_activations
_output_quantizers*
Õ
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses

 layer
¡optimizer_step
¢_weight_vars
£
kernel_min
¤
kernel_max
¥_quantize_activations
¦post_activation_min
§post_activation_max
¨_output_quantizers*
ÿ
©	variables
ªtrainable_variables
«regularization_losses
¬	keras_api
­__call__
+®&call_and_return_all_conditional_losses

¯layer
°optimizer_step
±_weight_vars
²_quantize_activations
³_output_quantizers*
Õ
´	variables
µtrainable_variables
¶regularization_losses
·	keras_api
¸__call__
+¹&call_and_return_all_conditional_losses

ºlayer
»optimizer_step
¼_weight_vars
½
kernel_min
¾
kernel_max
¿_quantize_activations
Àpost_activation_min
Ápost_activation_max
Â_output_quantizers*
ÿ
Ã	variables
Ätrainable_variables
Åregularization_losses
Æ	keras_api
Ç__call__
+È&call_and_return_all_conditional_losses

Élayer
Êoptimizer_step
Ë_weight_vars
Ì_quantize_activations
Í_output_quantizers*
Ó
Î	variables
Ïtrainable_variables
Ðregularization_losses
Ñ	keras_api
Ò__call__
+Ó&call_and_return_all_conditional_losses

Ôlayer
Õoptimizer_step
Ö_weight_vars
×
kernel_min
Ø
kernel_max
Ù_quantize_activations
Úpre_activation_min
Ûpre_activation_max
Ü_output_quantizers*
§
0
 1
"2
Ý3
Þ4
*5
,6
-7
/8
09
910
ß11
à12
D13
F14
G15
I16
J17
S18
á19
â20
^21
`22
a23
c24
d25
ã26
ä27
m28
o29
p30
r31
s32
å33
æ34
|35
~36
37
38
39
40
41
ç42
è43
¡44
£45
¤46
¦47
§48
°49
é50
ê51
»52
½53
¾54
À55
Á56
Ê57
ë58
ì59
Õ60
×61
Ø62
Ú63
Û64*

Ý0
Þ1
ß2
à3
á4
â5
ã6
ä7
å8
æ9
ç10
è11
é12
ê13
ë14
ì15*
* 
µ
ínon_trainable_variables
îlayers
ïmetrics
 ðlayer_regularization_losses
ñlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
:
òtrace_0
ótrace_1
ôtrace_2
õtrace_3* 
:
ötrace_0
÷trace_1
øtrace_2
ùtrace_3* 
* 
©
	úiter
ûbeta_1
übeta_2

ýdecay
þlearning_rate	Ým×	ÞmØ	ßmÙ	àmÚ	ámÛ	âmÜ	ãmÝ	ämÞ	åmß	æmà	çmá	èmâ	émã	êmä	ëmå	ìmæ	Ývç	Þvè	ßvé	àvê	ávë	âvì	ãví	ävî	åvï	ævð	çvñ	èvò	évó	êvô	ëvõ	ìvö*

ÿserving_default* 

0
 1
"2*
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

trace_0
trace_1* 

trace_0
trace_1* 
}w
VARIABLE_VALUE!quantize_layer/quantize_layer_minBlayer_with_weights-0/quantize_layer_min/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUE!quantize_layer/quantize_layer_maxBlayer_with_weights-0/quantize_layer_max/.ATTRIBUTES/VARIABLE_VALUE*

min_var
 max_var*
uo
VARIABLE_VALUEquantize_layer/optimizer_step>layer_with_weights-0/optimizer_step/.ATTRIBUTES/VARIABLE_VALUE*
7
Ý0
Þ1
*2
,3
-4
/5
06*

Ý0
Þ1*


0* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
#	variables
$trainable_variables
%regularization_losses
'__call__
*(&call_and_return_all_conditional_losses
&("call_and_return_conditional_losses*

trace_0
trace_1* 

trace_0
trace_1* 
Ñ
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
Ýkernel
	Þbias
!_jit_compiled_convolution_op*
sm
VARIABLE_VALUEquant_conv2d/optimizer_step>layer_with_weights-1/optimizer_step/.ATTRIBUTES/VARIABLE_VALUE*

0*
ke
VARIABLE_VALUEquant_conv2d/kernel_min:layer_with_weights-1/kernel_min/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUEquant_conv2d/kernel_max:layer_with_weights-1/kernel_max/.ATTRIBUTES/VARIABLE_VALUE*
* 
}w
VARIABLE_VALUE quant_conv2d/post_activation_minClayer_with_weights-1/post_activation_min/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUE quant_conv2d/post_activation_maxClayer_with_weights-1/post_activation_max/.ATTRIBUTES/VARIABLE_VALUE*
* 

90*
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
2	variables
3trainable_variables
4regularization_losses
6__call__
*7&call_and_return_all_conditional_losses
&7"call_and_return_conditional_losses*

 trace_0
¡trace_1* 

¢trace_0
£trace_1* 

¤	variables
¥trainable_variables
¦regularization_losses
§	keras_api
¨__call__
+©&call_and_return_all_conditional_losses* 
zt
VARIABLE_VALUE"quant_max_pooling2d/optimizer_step>layer_with_weights-2/optimizer_step/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
7
ß0
à1
D2
F3
G4
I5
J6*

ß0
à1*


ª0* 

«non_trainable_variables
¬layers
­metrics
 ®layer_regularization_losses
¯layer_metrics
=	variables
>trainable_variables
?regularization_losses
A__call__
*B&call_and_return_all_conditional_losses
&B"call_and_return_conditional_losses*

°trace_0
±trace_1* 

²trace_0
³trace_1* 
Ñ
´	variables
µtrainable_variables
¶regularization_losses
·	keras_api
¸__call__
+¹&call_and_return_all_conditional_losses
ßkernel
	àbias
!º_jit_compiled_convolution_op*
uo
VARIABLE_VALUEquant_conv2d_1/optimizer_step>layer_with_weights-3/optimizer_step/.ATTRIBUTES/VARIABLE_VALUE*

»0*
mg
VARIABLE_VALUEquant_conv2d_1/kernel_min:layer_with_weights-3/kernel_min/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUEquant_conv2d_1/kernel_max:layer_with_weights-3/kernel_max/.ATTRIBUTES/VARIABLE_VALUE*
* 
y
VARIABLE_VALUE"quant_conv2d_1/post_activation_minClayer_with_weights-3/post_activation_min/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE"quant_conv2d_1/post_activation_maxClayer_with_weights-3/post_activation_max/.ATTRIBUTES/VARIABLE_VALUE*
* 

S0*
* 
* 

¼non_trainable_variables
½layers
¾metrics
 ¿layer_regularization_losses
Àlayer_metrics
L	variables
Mtrainable_variables
Nregularization_losses
P__call__
*Q&call_and_return_all_conditional_losses
&Q"call_and_return_conditional_losses*

Átrace_0
Âtrace_1* 

Ãtrace_0
Ätrace_1* 

Å	variables
Ætrainable_variables
Çregularization_losses
È	keras_api
É__call__
+Ê&call_and_return_all_conditional_losses* 
|v
VARIABLE_VALUE$quant_max_pooling2d_1/optimizer_step>layer_with_weights-4/optimizer_step/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
7
á0
â1
^2
`3
a4
c5
d6*

á0
â1*


Ë0* 

Ìnon_trainable_variables
Ílayers
Îmetrics
 Ïlayer_regularization_losses
Ðlayer_metrics
W	variables
Xtrainable_variables
Yregularization_losses
[__call__
*\&call_and_return_all_conditional_losses
&\"call_and_return_conditional_losses*

Ñtrace_0
Òtrace_1* 

Ótrace_0
Ôtrace_1* 
Ñ
Õ	variables
Ötrainable_variables
×regularization_losses
Ø	keras_api
Ù__call__
+Ú&call_and_return_all_conditional_losses
ákernel
	âbias
!Û_jit_compiled_convolution_op*
uo
VARIABLE_VALUEquant_conv2d_2/optimizer_step>layer_with_weights-5/optimizer_step/.ATTRIBUTES/VARIABLE_VALUE*

Ü0*
mg
VARIABLE_VALUEquant_conv2d_2/kernel_min:layer_with_weights-5/kernel_min/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUEquant_conv2d_2/kernel_max:layer_with_weights-5/kernel_max/.ATTRIBUTES/VARIABLE_VALUE*
* 
y
VARIABLE_VALUE"quant_conv2d_2/post_activation_minClayer_with_weights-5/post_activation_min/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE"quant_conv2d_2/post_activation_maxClayer_with_weights-5/post_activation_max/.ATTRIBUTES/VARIABLE_VALUE*
* 
7
ã0
ä1
m2
o3
p4
r5
s6*

ã0
ä1*


Ý0* 

Þnon_trainable_variables
ßlayers
àmetrics
 álayer_regularization_losses
âlayer_metrics
f	variables
gtrainable_variables
hregularization_losses
j__call__
*k&call_and_return_all_conditional_losses
&k"call_and_return_conditional_losses*

ãtrace_0
ätrace_1* 

åtrace_0
ætrace_1* 
Ñ
ç	variables
ètrainable_variables
éregularization_losses
ê	keras_api
ë__call__
+ì&call_and_return_all_conditional_losses
ãkernel
	äbias
!í_jit_compiled_convolution_op*
uo
VARIABLE_VALUEquant_conv2d_3/optimizer_step>layer_with_weights-6/optimizer_step/.ATTRIBUTES/VARIABLE_VALUE*

î0*
mg
VARIABLE_VALUEquant_conv2d_3/kernel_min:layer_with_weights-6/kernel_min/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUEquant_conv2d_3/kernel_max:layer_with_weights-6/kernel_max/.ATTRIBUTES/VARIABLE_VALUE*
* 
y
VARIABLE_VALUE"quant_conv2d_3/post_activation_minClayer_with_weights-6/post_activation_min/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE"quant_conv2d_3/post_activation_maxClayer_with_weights-6/post_activation_max/.ATTRIBUTES/VARIABLE_VALUE*
* 
9
å0
æ1
|2
~3
4
5
6*

å0
æ1*


ï0* 

ðnon_trainable_variables
ñlayers
òmetrics
 ólayer_regularization_losses
ôlayer_metrics
u	variables
vtrainable_variables
wregularization_losses
y__call__
*z&call_and_return_all_conditional_losses
&z"call_and_return_conditional_losses*

õtrace_0
ötrace_1* 

÷trace_0
øtrace_1* 
Ñ
ù	variables
útrainable_variables
ûregularization_losses
ü	keras_api
ý__call__
+þ&call_and_return_all_conditional_losses
åkernel
	æbias
!ÿ_jit_compiled_convolution_op*
uo
VARIABLE_VALUEquant_conv2d_4/optimizer_step>layer_with_weights-7/optimizer_step/.ATTRIBUTES/VARIABLE_VALUE*

0*
mg
VARIABLE_VALUEquant_conv2d_4/kernel_min:layer_with_weights-7/kernel_min/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUEquant_conv2d_4/kernel_max:layer_with_weights-7/kernel_max/.ATTRIBUTES/VARIABLE_VALUE*
* 
y
VARIABLE_VALUE"quant_conv2d_4/post_activation_minClayer_with_weights-7/post_activation_min/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE"quant_conv2d_4/post_activation_maxClayer_with_weights-7/post_activation_max/.ATTRIBUTES/VARIABLE_VALUE*
* 

0*
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*

trace_0
trace_1* 

trace_0
trace_1* 

	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses* 
|v
VARIABLE_VALUE$quant_max_pooling2d_2/optimizer_step>layer_with_weights-8/optimizer_step/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 

0*
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*

trace_0
trace_1* 

trace_0
trace_1* 

	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses* 
tn
VARIABLE_VALUEquant_flatten/optimizer_step>layer_with_weights-9/optimizer_step/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
<
ç0
è1
¡2
£3
¤4
¦5
§6*

ç0
è1*


0* 

 non_trainable_variables
¡layers
¢metrics
 £layer_regularization_losses
¤layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*

¥trace_0
¦trace_1* 

§trace_0
¨trace_1* 
®
©	variables
ªtrainable_variables
«regularization_losses
¬	keras_api
­__call__
+®&call_and_return_all_conditional_losses
çkernel
	èbias*
sm
VARIABLE_VALUEquant_dense/optimizer_step?layer_with_weights-10/optimizer_step/.ATTRIBUTES/VARIABLE_VALUE*

¯0*
ke
VARIABLE_VALUEquant_dense/kernel_min;layer_with_weights-10/kernel_min/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUEquant_dense/kernel_max;layer_with_weights-10/kernel_max/.ATTRIBUTES/VARIABLE_VALUE*
* 
}w
VARIABLE_VALUEquant_dense/post_activation_minDlayer_with_weights-10/post_activation_min/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEquant_dense/post_activation_maxDlayer_with_weights-10/post_activation_max/.ATTRIBUTES/VARIABLE_VALUE*
* 

°0*
* 
* 

°non_trainable_variables
±layers
²metrics
 ³layer_regularization_losses
´layer_metrics
©	variables
ªtrainable_variables
«regularization_losses
­__call__
+®&call_and_return_all_conditional_losses
'®"call_and_return_conditional_losses*

µtrace_0
¶trace_1* 

·trace_0
¸trace_1* 
¬
¹	variables
ºtrainable_variables
»regularization_losses
¼	keras_api
½__call__
+¾&call_and_return_all_conditional_losses
¿_random_generator* 
uo
VARIABLE_VALUEquant_dropout/optimizer_step?layer_with_weights-11/optimizer_step/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
<
é0
ê1
»2
½3
¾4
À5
Á6*

é0
ê1*


À0* 

Ánon_trainable_variables
Âlayers
Ãmetrics
 Älayer_regularization_losses
Ålayer_metrics
´	variables
µtrainable_variables
¶regularization_losses
¸__call__
+¹&call_and_return_all_conditional_losses
'¹"call_and_return_conditional_losses*

Ætrace_0
Çtrace_1* 

Ètrace_0
Étrace_1* 
®
Ê	variables
Ëtrainable_variables
Ìregularization_losses
Í	keras_api
Î__call__
+Ï&call_and_return_all_conditional_losses
ékernel
	êbias*
uo
VARIABLE_VALUEquant_dense_1/optimizer_step?layer_with_weights-12/optimizer_step/.ATTRIBUTES/VARIABLE_VALUE*

Ð0*
mg
VARIABLE_VALUEquant_dense_1/kernel_min;layer_with_weights-12/kernel_min/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUEquant_dense_1/kernel_max;layer_with_weights-12/kernel_max/.ATTRIBUTES/VARIABLE_VALUE*
* 
y
VARIABLE_VALUE!quant_dense_1/post_activation_minDlayer_with_weights-12/post_activation_min/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE!quant_dense_1/post_activation_maxDlayer_with_weights-12/post_activation_max/.ATTRIBUTES/VARIABLE_VALUE*
* 

Ê0*
* 
* 

Ñnon_trainable_variables
Òlayers
Ómetrics
 Ôlayer_regularization_losses
Õlayer_metrics
Ã	variables
Ätrainable_variables
Åregularization_losses
Ç__call__
+È&call_and_return_all_conditional_losses
'È"call_and_return_conditional_losses*

Ötrace_0
×trace_1* 

Øtrace_0
Ùtrace_1* 
¬
Ú	variables
Ûtrainable_variables
Üregularization_losses
Ý	keras_api
Þ__call__
+ß&call_and_return_all_conditional_losses
à_random_generator* 
wq
VARIABLE_VALUEquant_dropout_1/optimizer_step?layer_with_weights-13/optimizer_step/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
<
ë0
ì1
Õ2
×3
Ø4
Ú5
Û6*

ë0
ì1*
* 

ánon_trainable_variables
âlayers
ãmetrics
 älayer_regularization_losses
ålayer_metrics
Î	variables
Ïtrainable_variables
Ðregularization_losses
Ò__call__
+Ó&call_and_return_all_conditional_losses
'Ó"call_and_return_conditional_losses*

ætrace_0
çtrace_1* 

ètrace_0
étrace_1* 
®
ê	variables
ëtrainable_variables
ìregularization_losses
í	keras_api
î__call__
+ï&call_and_return_all_conditional_losses
ëkernel
	ìbias*
uo
VARIABLE_VALUEquant_dense_2/optimizer_step?layer_with_weights-14/optimizer_step/.ATTRIBUTES/VARIABLE_VALUE*

ð0*
mg
VARIABLE_VALUEquant_dense_2/kernel_min;layer_with_weights-14/kernel_min/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUEquant_dense_2/kernel_max;layer_with_weights-14/kernel_max/.ATTRIBUTES/VARIABLE_VALUE*
* 
}w
VARIABLE_VALUE quant_dense_2/pre_activation_minClayer_with_weights-14/pre_activation_min/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUE quant_dense_2/pre_activation_maxClayer_with_weights-14/pre_activation_max/.ATTRIBUTES/VARIABLE_VALUE*
* 
MG
VARIABLE_VALUEconv2d/kernel&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
KE
VARIABLE_VALUEconv2d/bias&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEconv2d_1/kernel'variables/11/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUEconv2d_1/bias'variables/12/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEconv2d_2/kernel'variables/19/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUEconv2d_2/bias'variables/20/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEconv2d_3/kernel'variables/26/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUEconv2d_3/bias'variables/27/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEconv2d_4/kernel'variables/33/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUEconv2d_4/bias'variables/34/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEdense/kernel'variables/42/.ATTRIBUTES/VARIABLE_VALUE*
KE
VARIABLE_VALUE
dense/bias'variables/43/.ATTRIBUTES/VARIABLE_VALUE*
OI
VARIABLE_VALUEdense_1/kernel'variables/50/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEdense_1/bias'variables/51/.ATTRIBUTES/VARIABLE_VALUE*
OI
VARIABLE_VALUEdense_2/kernel'variables/58/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEdense_2/bias'variables/59/.ATTRIBUTES/VARIABLE_VALUE*

0
 1
"2
*3
,4
-5
/6
07
98
D9
F10
G11
I12
J13
S14
^15
`16
a17
c18
d19
m20
o21
p22
r23
s24
|25
~26
27
28
29
30
31
¡32
£33
¤34
¦35
§36
°37
»38
½39
¾40
À41
Á42
Ê43
Õ44
×45
Ø46
Ú47
Û48*
r
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14*

ñ0
ò1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 

0
 1
"2*
* 
* 
* 
* 
* 
* 
* 
* 

ótrace_0* 
'
*0
,1
-2
/3
04*

)0*
* 
* 
* 
* 
* 
* 
* 

Þ0*

Þ0*


0* 

ônon_trainable_variables
õlayers
ömetrics
 ÷layer_regularization_losses
ølayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*
* 
* 
* 

Ý0
ù2*

90*
	
80* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

únon_trainable_variables
ûlayers
ümetrics
 ýlayer_regularization_losses
þlayer_metrics
¤	variables
¥trainable_variables
¦regularization_losses
¨__call__
+©&call_and_return_all_conditional_losses
'©"call_and_return_conditional_losses* 

ÿtrace_0* 

trace_0* 

trace_0* 
'
D0
F1
G2
I3
J4*

C0*
* 
* 
* 
* 
* 
* 
* 

à0*

à0*


ª0* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
´	variables
µtrainable_variables
¶regularization_losses
¸__call__
+¹&call_and_return_all_conditional_losses
'¹"call_and_return_conditional_losses*
* 
* 
* 

ß0
2*

S0*
	
R0* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
Å	variables
Ætrainable_variables
Çregularization_losses
É__call__
+Ê&call_and_return_all_conditional_losses
'Ê"call_and_return_conditional_losses* 

trace_0* 

trace_0* 

trace_0* 
'
^0
`1
a2
c3
d4*

]0*
* 
* 
* 
* 
* 
* 
* 

â0*

â0*


Ë0* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
Õ	variables
Ötrainable_variables
×regularization_losses
Ù__call__
+Ú&call_and_return_all_conditional_losses
'Ú"call_and_return_conditional_losses*
* 
* 
* 

á0
2*

trace_0* 
'
m0
o1
p2
r3
s4*

l0*
* 
* 
* 
* 
* 
* 
* 

ä0*

ä0*


Ý0* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
ç	variables
ètrainable_variables
éregularization_losses
ë__call__
+ì&call_and_return_all_conditional_losses
'ì"call_and_return_conditional_losses*
* 
* 
* 

ã0
2*

trace_0* 
)
|0
~1
2
3
4*

{0*
* 
* 
* 
* 
* 
* 
* 

æ0*

æ0*


ï0* 

non_trainable_variables
layers
 metrics
 ¡layer_regularization_losses
¢layer_metrics
ù	variables
útrainable_variables
ûregularization_losses
ý__call__
+þ&call_and_return_all_conditional_losses
'þ"call_and_return_conditional_losses*
* 
* 
* 

å0
£2*

0*


0* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

¤non_trainable_variables
¥layers
¦metrics
 §layer_regularization_losses
¨layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses* 

©trace_0* 

ªtrace_0* 

0*


0* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

«non_trainable_variables
¬layers
­metrics
 ®layer_regularization_losses
¯layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses* 
* 
* 

°trace_0* 
,
¡0
£1
¤2
¦3
§4*

 0*
* 
* 
* 
* 
* 
* 
* 

è0*

è0*


0* 

±non_trainable_variables
²layers
³metrics
 ´layer_regularization_losses
µlayer_metrics
©	variables
ªtrainable_variables
«regularization_losses
­__call__
+®&call_and_return_all_conditional_losses
'®"call_and_return_conditional_losses*
* 
* 

ç0
¶2*

°0*


¯0* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

·non_trainable_variables
¸layers
¹metrics
 ºlayer_regularization_losses
»layer_metrics
¹	variables
ºtrainable_variables
»regularization_losses
½__call__
+¾&call_and_return_all_conditional_losses
'¾"call_and_return_conditional_losses* 
* 
* 
* 

¼trace_0* 
,
»0
½1
¾2
À3
Á4*

º0*
* 
* 
* 
* 
* 
* 
* 

ê0*

ê0*


À0* 

½non_trainable_variables
¾layers
¿metrics
 Àlayer_regularization_losses
Álayer_metrics
Ê	variables
Ëtrainable_variables
Ìregularization_losses
Î__call__
+Ï&call_and_return_all_conditional_losses
'Ï"call_and_return_conditional_losses*
* 
* 

é0
Â2*

Ê0*


É0* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

Ãnon_trainable_variables
Älayers
Åmetrics
 Ælayer_regularization_losses
Çlayer_metrics
Ú	variables
Ûtrainable_variables
Üregularization_losses
Þ__call__
+ß&call_and_return_all_conditional_losses
'ß"call_and_return_conditional_losses* 
* 
* 
* 
,
Õ0
×1
Ø2
Ú3
Û4*

Ô0*
* 
* 
* 
* 
* 
* 
* 

ì0*

ì0*
* 

Ènon_trainable_variables
Élayers
Êmetrics
 Ëlayer_regularization_losses
Ìlayer_metrics
ê	variables
ëtrainable_variables
ìregularization_losses
î__call__
+ï&call_and_return_all_conditional_losses
'ï"call_and_return_conditional_losses*
* 
* 

ë0
Í2*
<
Î	variables
Ï	keras_api

Ðtotal

Ñcount*
M
Ò	variables
Ó	keras_api

Ôtotal

Õcount
Ö
_fn_kwargs*
* 
* 
* 
* 


0* 
* 

,min_var
-max_var*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 


ª0* 
* 

Fmin_var
Gmax_var*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 


Ë0* 
* 

`min_var
amax_var*
* 
* 
* 
* 


Ý0* 
* 

omin_var
pmax_var*
* 
* 
* 
* 


ï0* 
* 

~min_var
max_var*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 


0* 
* 
 
£min_var
¤max_var*
* 
* 
* 
* 
* 
* 
* 
* 
* 


À0* 
* 
 
½min_var
¾max_var*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
 
×min_var
Ømax_var*

Ð0
Ñ1*

Î	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

Ô0
Õ1*

Ò	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
pj
VARIABLE_VALUEAdam/conv2d/kernel/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUEAdam/conv2d/bias/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUEAdam/conv2d_1/kernel/mCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUEAdam/conv2d_1/bias/mCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUEAdam/conv2d_2/kernel/mCvariables/19/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUEAdam/conv2d_2/bias/mCvariables/20/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUEAdam/conv2d_3/kernel/mCvariables/26/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUEAdam/conv2d_3/bias/mCvariables/27/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUEAdam/conv2d_4/kernel/mCvariables/33/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUEAdam/conv2d_4/bias/mCvariables/34/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUEAdam/dense/kernel/mCvariables/42/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUEAdam/dense/bias/mCvariables/43/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUEAdam/dense_1/kernel/mCvariables/50/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUEAdam/dense_1/bias/mCvariables/51/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUEAdam/dense_2/kernel/mCvariables/58/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUEAdam/dense_2/bias/mCvariables/59/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUEAdam/conv2d/kernel/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUEAdam/conv2d/bias/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUEAdam/conv2d_1/kernel/vCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUEAdam/conv2d_1/bias/vCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUEAdam/conv2d_2/kernel/vCvariables/19/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUEAdam/conv2d_2/bias/vCvariables/20/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUEAdam/conv2d_3/kernel/vCvariables/26/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUEAdam/conv2d_3/bias/vCvariables/27/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUEAdam/conv2d_4/kernel/vCvariables/33/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUEAdam/conv2d_4/bias/vCvariables/34/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUEAdam/dense/kernel/vCvariables/42/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUEAdam/dense/bias/vCvariables/43/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUEAdam/dense_1/kernel/vCvariables/50/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUEAdam/dense_1/bias/vCvariables/51/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUEAdam/dense_2/kernel/vCvariables/58/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUEAdam/dense_2/bias/vCvariables/59/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
ê'
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename5quantize_layer/quantize_layer_min/Read/ReadVariableOp5quantize_layer/quantize_layer_max/Read/ReadVariableOp1quantize_layer/optimizer_step/Read/ReadVariableOp/quant_conv2d/optimizer_step/Read/ReadVariableOp+quant_conv2d/kernel_min/Read/ReadVariableOp+quant_conv2d/kernel_max/Read/ReadVariableOp4quant_conv2d/post_activation_min/Read/ReadVariableOp4quant_conv2d/post_activation_max/Read/ReadVariableOp6quant_max_pooling2d/optimizer_step/Read/ReadVariableOp1quant_conv2d_1/optimizer_step/Read/ReadVariableOp-quant_conv2d_1/kernel_min/Read/ReadVariableOp-quant_conv2d_1/kernel_max/Read/ReadVariableOp6quant_conv2d_1/post_activation_min/Read/ReadVariableOp6quant_conv2d_1/post_activation_max/Read/ReadVariableOp8quant_max_pooling2d_1/optimizer_step/Read/ReadVariableOp1quant_conv2d_2/optimizer_step/Read/ReadVariableOp-quant_conv2d_2/kernel_min/Read/ReadVariableOp-quant_conv2d_2/kernel_max/Read/ReadVariableOp6quant_conv2d_2/post_activation_min/Read/ReadVariableOp6quant_conv2d_2/post_activation_max/Read/ReadVariableOp1quant_conv2d_3/optimizer_step/Read/ReadVariableOp-quant_conv2d_3/kernel_min/Read/ReadVariableOp-quant_conv2d_3/kernel_max/Read/ReadVariableOp6quant_conv2d_3/post_activation_min/Read/ReadVariableOp6quant_conv2d_3/post_activation_max/Read/ReadVariableOp1quant_conv2d_4/optimizer_step/Read/ReadVariableOp-quant_conv2d_4/kernel_min/Read/ReadVariableOp-quant_conv2d_4/kernel_max/Read/ReadVariableOp6quant_conv2d_4/post_activation_min/Read/ReadVariableOp6quant_conv2d_4/post_activation_max/Read/ReadVariableOp8quant_max_pooling2d_2/optimizer_step/Read/ReadVariableOp0quant_flatten/optimizer_step/Read/ReadVariableOp.quant_dense/optimizer_step/Read/ReadVariableOp*quant_dense/kernel_min/Read/ReadVariableOp*quant_dense/kernel_max/Read/ReadVariableOp3quant_dense/post_activation_min/Read/ReadVariableOp3quant_dense/post_activation_max/Read/ReadVariableOp0quant_dropout/optimizer_step/Read/ReadVariableOp0quant_dense_1/optimizer_step/Read/ReadVariableOp,quant_dense_1/kernel_min/Read/ReadVariableOp,quant_dense_1/kernel_max/Read/ReadVariableOp5quant_dense_1/post_activation_min/Read/ReadVariableOp5quant_dense_1/post_activation_max/Read/ReadVariableOp2quant_dropout_1/optimizer_step/Read/ReadVariableOp0quant_dense_2/optimizer_step/Read/ReadVariableOp,quant_dense_2/kernel_min/Read/ReadVariableOp,quant_dense_2/kernel_max/Read/ReadVariableOp4quant_dense_2/pre_activation_min/Read/ReadVariableOp4quant_dense_2/pre_activation_max/Read/ReadVariableOp!conv2d/kernel/Read/ReadVariableOpconv2d/bias/Read/ReadVariableOp#conv2d_1/kernel/Read/ReadVariableOp!conv2d_1/bias/Read/ReadVariableOp#conv2d_2/kernel/Read/ReadVariableOp!conv2d_2/bias/Read/ReadVariableOp#conv2d_3/kernel/Read/ReadVariableOp!conv2d_3/bias/Read/ReadVariableOp#conv2d_4/kernel/Read/ReadVariableOp!conv2d_4/bias/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOp"dense_2/kernel/Read/ReadVariableOp dense_2/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp(Adam/conv2d/kernel/m/Read/ReadVariableOp&Adam/conv2d/bias/m/Read/ReadVariableOp*Adam/conv2d_1/kernel/m/Read/ReadVariableOp(Adam/conv2d_1/bias/m/Read/ReadVariableOp*Adam/conv2d_2/kernel/m/Read/ReadVariableOp(Adam/conv2d_2/bias/m/Read/ReadVariableOp*Adam/conv2d_3/kernel/m/Read/ReadVariableOp(Adam/conv2d_3/bias/m/Read/ReadVariableOp*Adam/conv2d_4/kernel/m/Read/ReadVariableOp(Adam/conv2d_4/bias/m/Read/ReadVariableOp'Adam/dense/kernel/m/Read/ReadVariableOp%Adam/dense/bias/m/Read/ReadVariableOp)Adam/dense_1/kernel/m/Read/ReadVariableOp'Adam/dense_1/bias/m/Read/ReadVariableOp)Adam/dense_2/kernel/m/Read/ReadVariableOp'Adam/dense_2/bias/m/Read/ReadVariableOp(Adam/conv2d/kernel/v/Read/ReadVariableOp&Adam/conv2d/bias/v/Read/ReadVariableOp*Adam/conv2d_1/kernel/v/Read/ReadVariableOp(Adam/conv2d_1/bias/v/Read/ReadVariableOp*Adam/conv2d_2/kernel/v/Read/ReadVariableOp(Adam/conv2d_2/bias/v/Read/ReadVariableOp*Adam/conv2d_3/kernel/v/Read/ReadVariableOp(Adam/conv2d_3/bias/v/Read/ReadVariableOp*Adam/conv2d_4/kernel/v/Read/ReadVariableOp(Adam/conv2d_4/bias/v/Read/ReadVariableOp'Adam/dense/kernel/v/Read/ReadVariableOp%Adam/dense/bias/v/Read/ReadVariableOp)Adam/dense_1/kernel/v/Read/ReadVariableOp'Adam/dense_1/bias/v/Read/ReadVariableOp)Adam/dense_2/kernel/v/Read/ReadVariableOp'Adam/dense_2/bias/v/Read/ReadVariableOpConst*w
Tinp
n2l	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *;
config_proto+)

CPU

GPU2*0J
	
   E 8 *(
f#R!
__inference__traced_save_147569

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename!quantize_layer/quantize_layer_min!quantize_layer/quantize_layer_maxquantize_layer/optimizer_stepquant_conv2d/optimizer_stepquant_conv2d/kernel_minquant_conv2d/kernel_max quant_conv2d/post_activation_min quant_conv2d/post_activation_max"quant_max_pooling2d/optimizer_stepquant_conv2d_1/optimizer_stepquant_conv2d_1/kernel_minquant_conv2d_1/kernel_max"quant_conv2d_1/post_activation_min"quant_conv2d_1/post_activation_max$quant_max_pooling2d_1/optimizer_stepquant_conv2d_2/optimizer_stepquant_conv2d_2/kernel_minquant_conv2d_2/kernel_max"quant_conv2d_2/post_activation_min"quant_conv2d_2/post_activation_maxquant_conv2d_3/optimizer_stepquant_conv2d_3/kernel_minquant_conv2d_3/kernel_max"quant_conv2d_3/post_activation_min"quant_conv2d_3/post_activation_maxquant_conv2d_4/optimizer_stepquant_conv2d_4/kernel_minquant_conv2d_4/kernel_max"quant_conv2d_4/post_activation_min"quant_conv2d_4/post_activation_max$quant_max_pooling2d_2/optimizer_stepquant_flatten/optimizer_stepquant_dense/optimizer_stepquant_dense/kernel_minquant_dense/kernel_maxquant_dense/post_activation_minquant_dense/post_activation_maxquant_dropout/optimizer_stepquant_dense_1/optimizer_stepquant_dense_1/kernel_minquant_dense_1/kernel_max!quant_dense_1/post_activation_min!quant_dense_1/post_activation_maxquant_dropout_1/optimizer_stepquant_dense_2/optimizer_stepquant_dense_2/kernel_minquant_dense_2/kernel_max quant_dense_2/pre_activation_min quant_dense_2/pre_activation_maxconv2d/kernelconv2d/biasconv2d_1/kernelconv2d_1/biasconv2d_2/kernelconv2d_2/biasconv2d_3/kernelconv2d_3/biasconv2d_4/kernelconv2d_4/biasdense/kernel
dense/biasdense_1/kerneldense_1/biasdense_2/kerneldense_2/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotal_1count_1totalcountAdam/conv2d/kernel/mAdam/conv2d/bias/mAdam/conv2d_1/kernel/mAdam/conv2d_1/bias/mAdam/conv2d_2/kernel/mAdam/conv2d_2/bias/mAdam/conv2d_3/kernel/mAdam/conv2d_3/bias/mAdam/conv2d_4/kernel/mAdam/conv2d_4/bias/mAdam/dense/kernel/mAdam/dense/bias/mAdam/dense_1/kernel/mAdam/dense_1/bias/mAdam/dense_2/kernel/mAdam/dense_2/bias/mAdam/conv2d/kernel/vAdam/conv2d/bias/vAdam/conv2d_1/kernel/vAdam/conv2d_1/bias/vAdam/conv2d_2/kernel/vAdam/conv2d_2/bias/vAdam/conv2d_3/kernel/vAdam/conv2d_3/bias/vAdam/conv2d_4/kernel/vAdam/conv2d_4/bias/vAdam/dense/kernel/vAdam/dense/bias/vAdam/dense_1/kernel/vAdam/dense_1/bias/vAdam/dense_2/kernel/vAdam/dense_2/bias/v*v
Tino
m2k*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *;
config_proto+)

CPU

GPU2*0J
	
   E 8 *+
f&R$
"__inference__traced_restore_147897·½*

Ù

$__inference_signature_wrapper_145187
conv2d_input
unknown: 
	unknown_0: #
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: 
	unknown_5: 
	unknown_6: #
	unknown_7:  
	unknown_8: 
	unknown_9: 

unknown_10: 

unknown_11: 

unknown_12: $

unknown_13: @

unknown_14:@

unknown_15:@

unknown_16:@

unknown_17: 

unknown_18: $

unknown_19:@@

unknown_20:@

unknown_21:@

unknown_22:@

unknown_23: 

unknown_24: $

unknown_25:@ 

unknown_26: 

unknown_27: 

unknown_28: 

unknown_29: 

unknown_30: 

unknown_31:


unknown_32: 

unknown_33: 

unknown_34:	

unknown_35: 

unknown_36: 

unknown_37:


unknown_38: 

unknown_39: 

unknown_40:	

unknown_41: 

unknown_42: 

unknown_43:	

unknown_44: 

unknown_45: 

unknown_46:

unknown_47: 

unknown_48: 
identity¢StatefulPartitionedCallá
StatefulPartitionedCallStatefulPartitionedCallconv2d_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48*>
Tin7
523*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*T
_read_only_resource_inputs6
42	
 !"#$%&'()*+,-./012*;
config_proto+)

CPU

GPU2*0J
	
   E 8 **
f%R#
!__inference__wrapped_model_142898o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:ÿÿÿÿÿÿÿÿÿææ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿææ
&
_user_specified_nameconv2d_input
Á&

G__inference_quant_dense_layer_call_and_return_conditional_losses_146804

inputsR
>lastvaluequant_fakequantwithminmaxvars_readvariableop_resource:
J
@lastvaluequant_fakequantwithminmaxvars_readvariableop_1_resource: J
@lastvaluequant_fakequantwithminmaxvars_readvariableop_2_resource: .
biasadd_readvariableop_resource:	K
Amovingavgquantize_fakequantwithminmaxvars_readvariableop_resource: M
Cmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: 
identity¢BiasAdd/ReadVariableOp¢5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp¢7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1¢7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2¢8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp¢:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1¢.dense/kernel/Regularizer/L2Loss/ReadVariableOp¶
5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp>lastvaluequant_fakequantwithminmaxvars_readvariableop_resource* 
_output_shapes
:
*
dtype0°
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp@lastvaluequant_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0°
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2ReadVariableOp@lastvaluequant_fakequantwithminmaxvars_readvariableop_2_resource*
_output_shapes
: *
dtype0¸
&LastValueQuant/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars=LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp:value:0?LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0?LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2:value:0* 
_output_shapes
:
*
narrow_range(}
MatMulMatMulinputs0LastValueQuant/FakeQuantWithMinMaxVars:outputs:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ²
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpAmovingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype0¶
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpCmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0
)MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsRelu:activations:0@MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BMovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¯
.dense/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp>lastvaluequant_fakequantwithminmaxvars_readvariableop_resource* 
_output_shapes
:
*
dtype0
dense/kernel/Regularizer/L2LossL2Loss6dense/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: c
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0(dense/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: 
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´
NoOpNoOp^BiasAdd/ReadVariableOp6^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp8^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_18^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_29^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1/^dense/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ: : : : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2n
5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp2r
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_17LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_12r
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_27LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_22t
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12`
.dense/kernel/Regularizer/L2Loss/ReadVariableOp.dense/kernel/Regularizer/L2Loss/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
°	
¹
__inference_loss_fn_4_147200T
:conv2d_4_kernel_regularizer_l2loss_readvariableop_resource:@ 
identity¢1conv2d_4/kernel/Regularizer/L2Loss/ReadVariableOp´
1conv2d_4/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp:conv2d_4_kernel_regularizer_l2loss_readvariableop_resource*&
_output_shapes
:@ *
dtype0
"conv2d_4/kernel/Regularizer/L2LossL2Loss9conv2d_4/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!conv2d_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o; 
conv2d_4/kernel/Regularizer/mulMul*conv2d_4/kernel/Regularizer/mul/x:output:0+conv2d_4/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: a
IdentityIdentity#conv2d_4/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: z
NoOpNoOp2^conv2d_4/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2f
1conv2d_4/kernel/Regularizer/L2Loss/ReadVariableOp1conv2d_4/kernel/Regularizer/L2Loss/ReadVariableOp
­X
º	
J__inference_quant_conv2d_3_layer_call_and_return_conditional_losses_146583

inputsI
/lastvaluequant_batchmin_readvariableop_resource:@@3
%lastvaluequant_assignminlast_resource:@3
%lastvaluequant_assignmaxlast_resource:@-
biasadd_readvariableop_resource:@@
6movingavgquantize_assignminema_readvariableop_resource: @
6movingavgquantize_assignmaxema_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢LastValueQuant/AssignMaxLast¢LastValueQuant/AssignMinLast¢&LastValueQuant/BatchMax/ReadVariableOp¢&LastValueQuant/BatchMin/ReadVariableOp¢?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp¢ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1¢ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2¢2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp¢-MovingAvgQuantize/AssignMaxEma/ReadVariableOp¢2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp¢-MovingAvgQuantize/AssignMinEma/ReadVariableOp¢8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp¢:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1¢1conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOp
&LastValueQuant/BatchMin/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:@@*
dtype0~
)LastValueQuant/BatchMin/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          §
LastValueQuant/BatchMinMin.LastValueQuant/BatchMin/ReadVariableOp:value:02LastValueQuant/BatchMin/reduction_indices:output:0*
T0*
_output_shapes
:@
&LastValueQuant/BatchMax/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:@@*
dtype0~
)LastValueQuant/BatchMax/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          §
LastValueQuant/BatchMaxMax.LastValueQuant/BatchMax/ReadVariableOp:value:02LastValueQuant/BatchMax/reduction_indices:output:0*
T0*
_output_shapes
:@]
LastValueQuant/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ¿
LastValueQuant/truedivRealDiv LastValueQuant/BatchMax:output:0!LastValueQuant/truediv/y:output:0*
T0*
_output_shapes
:@
LastValueQuant/MinimumMinimum LastValueQuant/BatchMin:output:0LastValueQuant/truediv:z:0*
T0*
_output_shapes
:@Y
LastValueQuant/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ¿
LastValueQuant/mulMul LastValueQuant/BatchMin:output:0LastValueQuant/mul/y:output:0*
T0*
_output_shapes
:@
LastValueQuant/MaximumMaximum LastValueQuant/BatchMax:output:0LastValueQuant/mul:z:0*
T0*
_output_shapes
:@¬
LastValueQuant/AssignMinLastAssignVariableOp%lastvaluequant_assignminlast_resourceLastValueQuant/Minimum:z:0*
_output_shapes
 *
dtype0*
validate_shape(¬
LastValueQuant/AssignMaxLastAssignVariableOp%lastvaluequant_assignmaxlast_resourceLastValueQuant/Maximum:z:0*
_output_shapes
 *
dtype0*
validate_shape(·
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:@@*
dtype0Â
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ReadVariableOp%lastvaluequant_assignminlast_resource^LastValueQuant/AssignMinLast*
_output_shapes
:@*
dtype0Â
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ReadVariableOp%lastvaluequant_assignmaxlast_resource^LastValueQuant/AssignMaxLast*
_output_shapes
:@*
dtype0ð
0LastValueQuant/FakeQuantWithMinMaxVarsPerChannel!FakeQuantWithMinMaxVarsPerChannelGLastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2:value:0*&
_output_shapes
:@@*
narrow_range(¶
Conv2DConv2Dinputs:LastValueQuant/FakeQuantWithMinMaxVarsPerChannel:outputs:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@p
MovingAvgQuantize/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             x
MovingAvgQuantize/BatchMinMinRelu:activations:0 MovingAvgQuantize/Const:output:0*
T0*
_output_shapes
: r
MovingAvgQuantize/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             z
MovingAvgQuantize/BatchMaxMaxRelu:activations:0"MovingAvgQuantize/Const_1:output:0*
T0*
_output_shapes
: `
MovingAvgQuantize/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
MovingAvgQuantize/MinimumMinimum#MovingAvgQuantize/BatchMin:output:0$MovingAvgQuantize/Minimum/y:output:0*
T0*
_output_shapes
: `
MovingAvgQuantize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
MovingAvgQuantize/MaximumMaximum#MovingAvgQuantize/BatchMax:output:0$MovingAvgQuantize/Maximum/y:output:0*
T0*
_output_shapes
: i
$MovingAvgQuantize/AssignMinEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o:
-MovingAvgQuantize/AssignMinEma/ReadVariableOpReadVariableOp6movingavgquantize_assignminema_readvariableop_resource*
_output_shapes
: *
dtype0 
"MovingAvgQuantize/AssignMinEma/subSub5MovingAvgQuantize/AssignMinEma/ReadVariableOp:value:0MovingAvgQuantize/Minimum:z:0*
T0*
_output_shapes
: ¡
"MovingAvgQuantize/AssignMinEma/mulMul&MovingAvgQuantize/AssignMinEma/sub:z:0-MovingAvgQuantize/AssignMinEma/decay:output:0*
T0*
_output_shapes
: ü
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOpAssignSubVariableOp6movingavgquantize_assignminema_readvariableop_resource&MovingAvgQuantize/AssignMinEma/mul:z:0.^MovingAvgQuantize/AssignMinEma/ReadVariableOp*
_output_shapes
 *
dtype0i
$MovingAvgQuantize/AssignMaxEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o:
-MovingAvgQuantize/AssignMaxEma/ReadVariableOpReadVariableOp6movingavgquantize_assignmaxema_readvariableop_resource*
_output_shapes
: *
dtype0 
"MovingAvgQuantize/AssignMaxEma/subSub5MovingAvgQuantize/AssignMaxEma/ReadVariableOp:value:0MovingAvgQuantize/Maximum:z:0*
T0*
_output_shapes
: ¡
"MovingAvgQuantize/AssignMaxEma/mulMul&MovingAvgQuantize/AssignMaxEma/sub:z:0-MovingAvgQuantize/AssignMaxEma/decay:output:0*
T0*
_output_shapes
: ü
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpAssignSubVariableOp6movingavgquantize_assignmaxema_readvariableop_resource&MovingAvgQuantize/AssignMaxEma/mul:z:0.^MovingAvgQuantize/AssignMaxEma/ReadVariableOp*
_output_shapes
 *
dtype0Ü
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp6movingavgquantize_assignminema_readvariableop_resource3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp*
_output_shapes
: *
dtype0Þ
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp6movingavgquantize_assignmaxema_readvariableop_resource3^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp*
_output_shapes
: *
dtype0
)MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsRelu:activations:0@MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BMovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@©
1conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:@@*
dtype0
"conv2d_3/kernel/Regularizer/L2LossL2Loss9conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!conv2d_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o; 
conv2d_3/kernel/Regularizer/mulMul*conv2d_3/kernel/Regularizer/mul/x:output:0+conv2d_3/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: 
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¯
NoOpNoOp^BiasAdd/ReadVariableOp^LastValueQuant/AssignMaxLast^LastValueQuant/AssignMinLast'^LastValueQuant/BatchMax/ReadVariableOp'^LastValueQuant/BatchMin/ReadVariableOp@^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpB^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1B^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_23^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp.^MovingAvgQuantize/AssignMaxEma/ReadVariableOp3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp.^MovingAvgQuantize/AssignMinEma/ReadVariableOp9^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12^conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ@: : : : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2<
LastValueQuant/AssignMaxLastLastValueQuant/AssignMaxLast2<
LastValueQuant/AssignMinLastLastValueQuant/AssignMinLast2P
&LastValueQuant/BatchMax/ReadVariableOp&LastValueQuant/BatchMax/ReadVariableOp2P
&LastValueQuant/BatchMin/ReadVariableOp&LastValueQuant/BatchMin/ReadVariableOp2
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp2
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_12
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_22h
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2^
-MovingAvgQuantize/AssignMaxEma/ReadVariableOp-MovingAvgQuantize/AssignMaxEma/ReadVariableOp2h
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2^
-MovingAvgQuantize/AssignMinEma/ReadVariableOp-MovingAvgQuantize/AssignMinEma/ReadVariableOp2t
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12f
1conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOp1conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
¤X
¶	
H__inference_quant_conv2d_layer_call_and_return_conditional_losses_146195

inputsI
/lastvaluequant_batchmin_readvariableop_resource: 3
%lastvaluequant_assignminlast_resource: 3
%lastvaluequant_assignmaxlast_resource: -
biasadd_readvariableop_resource: @
6movingavgquantize_assignminema_readvariableop_resource: @
6movingavgquantize_assignmaxema_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢LastValueQuant/AssignMaxLast¢LastValueQuant/AssignMinLast¢&LastValueQuant/BatchMax/ReadVariableOp¢&LastValueQuant/BatchMin/ReadVariableOp¢?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp¢ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1¢ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2¢2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp¢-MovingAvgQuantize/AssignMaxEma/ReadVariableOp¢2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp¢-MovingAvgQuantize/AssignMinEma/ReadVariableOp¢8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp¢:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1¢/conv2d/kernel/Regularizer/L2Loss/ReadVariableOp
&LastValueQuant/BatchMin/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
: *
dtype0~
)LastValueQuant/BatchMin/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          §
LastValueQuant/BatchMinMin.LastValueQuant/BatchMin/ReadVariableOp:value:02LastValueQuant/BatchMin/reduction_indices:output:0*
T0*
_output_shapes
: 
&LastValueQuant/BatchMax/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
: *
dtype0~
)LastValueQuant/BatchMax/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          §
LastValueQuant/BatchMaxMax.LastValueQuant/BatchMax/ReadVariableOp:value:02LastValueQuant/BatchMax/reduction_indices:output:0*
T0*
_output_shapes
: ]
LastValueQuant/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ¿
LastValueQuant/truedivRealDiv LastValueQuant/BatchMax:output:0!LastValueQuant/truediv/y:output:0*
T0*
_output_shapes
: 
LastValueQuant/MinimumMinimum LastValueQuant/BatchMin:output:0LastValueQuant/truediv:z:0*
T0*
_output_shapes
: Y
LastValueQuant/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ¿
LastValueQuant/mulMul LastValueQuant/BatchMin:output:0LastValueQuant/mul/y:output:0*
T0*
_output_shapes
: 
LastValueQuant/MaximumMaximum LastValueQuant/BatchMax:output:0LastValueQuant/mul:z:0*
T0*
_output_shapes
: ¬
LastValueQuant/AssignMinLastAssignVariableOp%lastvaluequant_assignminlast_resourceLastValueQuant/Minimum:z:0*
_output_shapes
 *
dtype0*
validate_shape(¬
LastValueQuant/AssignMaxLastAssignVariableOp%lastvaluequant_assignmaxlast_resourceLastValueQuant/Maximum:z:0*
_output_shapes
 *
dtype0*
validate_shape(·
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
: *
dtype0Â
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ReadVariableOp%lastvaluequant_assignminlast_resource^LastValueQuant/AssignMinLast*
_output_shapes
: *
dtype0Â
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ReadVariableOp%lastvaluequant_assignmaxlast_resource^LastValueQuant/AssignMaxLast*
_output_shapes
: *
dtype0ð
0LastValueQuant/FakeQuantWithMinMaxVarsPerChannel!FakeQuantWithMinMaxVarsPerChannelGLastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2:value:0*&
_output_shapes
: *
narrow_range(¹
Conv2DConv2Dinputs:LastValueQuant/FakeQuantWithMinMaxVarsPerChannel:outputs:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿââ *
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿââ Z
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿââ p
MovingAvgQuantize/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             x
MovingAvgQuantize/BatchMinMinRelu:activations:0 MovingAvgQuantize/Const:output:0*
T0*
_output_shapes
: r
MovingAvgQuantize/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             z
MovingAvgQuantize/BatchMaxMaxRelu:activations:0"MovingAvgQuantize/Const_1:output:0*
T0*
_output_shapes
: `
MovingAvgQuantize/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
MovingAvgQuantize/MinimumMinimum#MovingAvgQuantize/BatchMin:output:0$MovingAvgQuantize/Minimum/y:output:0*
T0*
_output_shapes
: `
MovingAvgQuantize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
MovingAvgQuantize/MaximumMaximum#MovingAvgQuantize/BatchMax:output:0$MovingAvgQuantize/Maximum/y:output:0*
T0*
_output_shapes
: i
$MovingAvgQuantize/AssignMinEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o:
-MovingAvgQuantize/AssignMinEma/ReadVariableOpReadVariableOp6movingavgquantize_assignminema_readvariableop_resource*
_output_shapes
: *
dtype0 
"MovingAvgQuantize/AssignMinEma/subSub5MovingAvgQuantize/AssignMinEma/ReadVariableOp:value:0MovingAvgQuantize/Minimum:z:0*
T0*
_output_shapes
: ¡
"MovingAvgQuantize/AssignMinEma/mulMul&MovingAvgQuantize/AssignMinEma/sub:z:0-MovingAvgQuantize/AssignMinEma/decay:output:0*
T0*
_output_shapes
: ü
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOpAssignSubVariableOp6movingavgquantize_assignminema_readvariableop_resource&MovingAvgQuantize/AssignMinEma/mul:z:0.^MovingAvgQuantize/AssignMinEma/ReadVariableOp*
_output_shapes
 *
dtype0i
$MovingAvgQuantize/AssignMaxEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o:
-MovingAvgQuantize/AssignMaxEma/ReadVariableOpReadVariableOp6movingavgquantize_assignmaxema_readvariableop_resource*
_output_shapes
: *
dtype0 
"MovingAvgQuantize/AssignMaxEma/subSub5MovingAvgQuantize/AssignMaxEma/ReadVariableOp:value:0MovingAvgQuantize/Maximum:z:0*
T0*
_output_shapes
: ¡
"MovingAvgQuantize/AssignMaxEma/mulMul&MovingAvgQuantize/AssignMaxEma/sub:z:0-MovingAvgQuantize/AssignMaxEma/decay:output:0*
T0*
_output_shapes
: ü
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpAssignSubVariableOp6movingavgquantize_assignmaxema_readvariableop_resource&MovingAvgQuantize/AssignMaxEma/mul:z:0.^MovingAvgQuantize/AssignMaxEma/ReadVariableOp*
_output_shapes
 *
dtype0Ü
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp6movingavgquantize_assignminema_readvariableop_resource3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp*
_output_shapes
: *
dtype0Þ
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp6movingavgquantize_assignmaxema_readvariableop_resource3^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp*
_output_shapes
: *
dtype0
)MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsRelu:activations:0@MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BMovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿââ §
/conv2d/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
: *
dtype0
 conv2d/kernel/Regularizer/L2LossL2Loss7conv2d/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: d
conv2d/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;
conv2d/kernel/Regularizer/mulMul(conv2d/kernel/Regularizer/mul/x:output:0)conv2d/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: 
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿââ ­
NoOpNoOp^BiasAdd/ReadVariableOp^LastValueQuant/AssignMaxLast^LastValueQuant/AssignMinLast'^LastValueQuant/BatchMax/ReadVariableOp'^LastValueQuant/BatchMin/ReadVariableOp@^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpB^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1B^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_23^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp.^MovingAvgQuantize/AssignMaxEma/ReadVariableOp3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp.^MovingAvgQuantize/AssignMinEma/ReadVariableOp9^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_10^conv2d/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):ÿÿÿÿÿÿÿÿÿææ: : : : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2<
LastValueQuant/AssignMaxLastLastValueQuant/AssignMaxLast2<
LastValueQuant/AssignMinLastLastValueQuant/AssignMinLast2P
&LastValueQuant/BatchMax/ReadVariableOp&LastValueQuant/BatchMax/ReadVariableOp2P
&LastValueQuant/BatchMin/ReadVariableOp&LastValueQuant/BatchMin/ReadVariableOp2
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp2
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_12
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_22h
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2^
-MovingAvgQuantize/AssignMaxEma/ReadVariableOp-MovingAvgQuantize/AssignMaxEma/ReadVariableOp2h
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2^
-MovingAvgQuantize/AssignMinEma/ReadVariableOp-MovingAvgQuantize/AssignMinEma/ReadVariableOp2t
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12b
/conv2d/kernel/Regularizer/L2Loss/ReadVariableOp/conv2d/kernel/Regularizer/L2Loss/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿææ
 
_user_specified_nameinputs
ÿ)
Û
J__inference_quant_conv2d_4_layer_call_and_return_conditional_losses_146646

inputsb
Hlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource:@ X
Jlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource: X
Jlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource: -
biasadd_readvariableop_resource: K
Amovingavgquantize_fakequantwithminmaxvars_readvariableop_resource: M
Cmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: 
identity¢BiasAdd/ReadVariableOp¢?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp¢ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1¢ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2¢8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp¢:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1¢1conv2d_4/kernel/Regularizer/L2Loss/ReadVariableOpÐ
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpReadVariableOpHlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource*&
_output_shapes
:@ *
dtype0È
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ReadVariableOpJlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource*
_output_shapes
: *
dtype0È
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ReadVariableOpJlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource*
_output_shapes
: *
dtype0ð
0LastValueQuant/FakeQuantWithMinMaxVarsPerChannel!FakeQuantWithMinMaxVarsPerChannelGLastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2:value:0*&
_output_shapes
:@ *
narrow_range(¶
Conv2DConv2Dinputs:LastValueQuant/FakeQuantWithMinMaxVarsPerChannel:outputs:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ²
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpAmovingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype0¶
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpCmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0
)MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsRelu:activations:0@MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BMovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Â
1conv2d_4/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpHlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource*&
_output_shapes
:@ *
dtype0
"conv2d_4/kernel/Regularizer/L2LossL2Loss9conv2d_4/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!conv2d_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o; 
conv2d_4/kernel/Regularizer/mulMul*conv2d_4/kernel/Regularizer/mul/x:output:0+conv2d_4/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: 
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Õ
NoOpNoOp^BiasAdd/ReadVariableOp@^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpB^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1B^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_29^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12^conv2d_4/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ@: : : : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp2
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_12
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_22t
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12f
1conv2d_4/kernel/Regularizer/L2Loss/ReadVariableOp1conv2d_4/kernel/Regularizer/L2Loss/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
	

/__inference_quant_conv2d_3_layer_call_fn_146488

inputs!
unknown:@@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3: 
	unknown_4: 
identity¢StatefulPartitionedCall©
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*(
_read_only_resource_inputs

*;
config_proto+)

CPU

GPU2*0J
	
   E 8 *S
fNRL
J__inference_quant_conv2d_3_layer_call_and_return_conditional_losses_143112w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ@: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
­
m
Q__inference_quant_max_pooling2d_2_layer_call_and_return_conditional_losses_143170

inputs
identity
MaxPoolMaxPoolinputs*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
ksize
*
paddingVALID*
strides
`
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ :W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
ÿ)
Û
J__inference_quant_conv2d_1_layer_call_and_return_conditional_losses_143027

inputsb
Hlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource:  X
Jlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource: X
Jlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource: -
biasadd_readvariableop_resource: K
Amovingavgquantize_fakequantwithminmaxvars_readvariableop_resource: M
Cmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: 
identity¢BiasAdd/ReadVariableOp¢?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp¢ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1¢ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2¢8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp¢:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1¢1conv2d_1/kernel/Regularizer/L2Loss/ReadVariableOpÐ
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpReadVariableOpHlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource*&
_output_shapes
:  *
dtype0È
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ReadVariableOpJlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource*
_output_shapes
: *
dtype0È
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ReadVariableOpJlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource*
_output_shapes
: *
dtype0ð
0LastValueQuant/FakeQuantWithMinMaxVarsPerChannel!FakeQuantWithMinMaxVarsPerChannelGLastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2:value:0*&
_output_shapes
:  *
narrow_range(¶
Conv2DConv2Dinputs:LastValueQuant/FakeQuantWithMinMaxVarsPerChannel:outputs:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿKK *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿKK X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿKK ²
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpAmovingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype0¶
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpCmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0
)MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsRelu:activations:0@MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BMovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿKK Â
1conv2d_1/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpHlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource*&
_output_shapes
:  *
dtype0
"conv2d_1/kernel/Regularizer/L2LossL2Loss9conv2d_1/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!conv2d_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o; 
conv2d_1/kernel/Regularizer/mulMul*conv2d_1/kernel/Regularizer/mul/x:output:0+conv2d_1/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: 
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿKK Õ
NoOpNoOp^BiasAdd/ReadVariableOp@^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpB^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1B^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_29^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12^conv2d_1/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿKK : : : : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp2
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_12
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_22t
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12f
1conv2d_1/kernel/Regularizer/L2Loss/ReadVariableOp1conv2d_1/kernel/Regularizer/L2Loss/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿKK 
 
_user_specified_nameinputs
­X
º	
J__inference_quant_conv2d_2_layer_call_and_return_conditional_losses_144027

inputsI
/lastvaluequant_batchmin_readvariableop_resource: @3
%lastvaluequant_assignminlast_resource:@3
%lastvaluequant_assignmaxlast_resource:@-
biasadd_readvariableop_resource:@@
6movingavgquantize_assignminema_readvariableop_resource: @
6movingavgquantize_assignmaxema_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢LastValueQuant/AssignMaxLast¢LastValueQuant/AssignMinLast¢&LastValueQuant/BatchMax/ReadVariableOp¢&LastValueQuant/BatchMin/ReadVariableOp¢?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp¢ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1¢ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2¢2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp¢-MovingAvgQuantize/AssignMaxEma/ReadVariableOp¢2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp¢-MovingAvgQuantize/AssignMinEma/ReadVariableOp¢8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp¢:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1¢1conv2d_2/kernel/Regularizer/L2Loss/ReadVariableOp
&LastValueQuant/BatchMin/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
: @*
dtype0~
)LastValueQuant/BatchMin/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          §
LastValueQuant/BatchMinMin.LastValueQuant/BatchMin/ReadVariableOp:value:02LastValueQuant/BatchMin/reduction_indices:output:0*
T0*
_output_shapes
:@
&LastValueQuant/BatchMax/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
: @*
dtype0~
)LastValueQuant/BatchMax/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          §
LastValueQuant/BatchMaxMax.LastValueQuant/BatchMax/ReadVariableOp:value:02LastValueQuant/BatchMax/reduction_indices:output:0*
T0*
_output_shapes
:@]
LastValueQuant/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ¿
LastValueQuant/truedivRealDiv LastValueQuant/BatchMax:output:0!LastValueQuant/truediv/y:output:0*
T0*
_output_shapes
:@
LastValueQuant/MinimumMinimum LastValueQuant/BatchMin:output:0LastValueQuant/truediv:z:0*
T0*
_output_shapes
:@Y
LastValueQuant/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ¿
LastValueQuant/mulMul LastValueQuant/BatchMin:output:0LastValueQuant/mul/y:output:0*
T0*
_output_shapes
:@
LastValueQuant/MaximumMaximum LastValueQuant/BatchMax:output:0LastValueQuant/mul:z:0*
T0*
_output_shapes
:@¬
LastValueQuant/AssignMinLastAssignVariableOp%lastvaluequant_assignminlast_resourceLastValueQuant/Minimum:z:0*
_output_shapes
 *
dtype0*
validate_shape(¬
LastValueQuant/AssignMaxLastAssignVariableOp%lastvaluequant_assignmaxlast_resourceLastValueQuant/Maximum:z:0*
_output_shapes
 *
dtype0*
validate_shape(·
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
: @*
dtype0Â
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ReadVariableOp%lastvaluequant_assignminlast_resource^LastValueQuant/AssignMinLast*
_output_shapes
:@*
dtype0Â
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ReadVariableOp%lastvaluequant_assignmaxlast_resource^LastValueQuant/AssignMaxLast*
_output_shapes
:@*
dtype0ð
0LastValueQuant/FakeQuantWithMinMaxVarsPerChannel!FakeQuantWithMinMaxVarsPerChannelGLastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2:value:0*&
_output_shapes
: @*
narrow_range(¶
Conv2DConv2Dinputs:LastValueQuant/FakeQuantWithMinMaxVarsPerChannel:outputs:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@p
MovingAvgQuantize/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             x
MovingAvgQuantize/BatchMinMinRelu:activations:0 MovingAvgQuantize/Const:output:0*
T0*
_output_shapes
: r
MovingAvgQuantize/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             z
MovingAvgQuantize/BatchMaxMaxRelu:activations:0"MovingAvgQuantize/Const_1:output:0*
T0*
_output_shapes
: `
MovingAvgQuantize/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
MovingAvgQuantize/MinimumMinimum#MovingAvgQuantize/BatchMin:output:0$MovingAvgQuantize/Minimum/y:output:0*
T0*
_output_shapes
: `
MovingAvgQuantize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
MovingAvgQuantize/MaximumMaximum#MovingAvgQuantize/BatchMax:output:0$MovingAvgQuantize/Maximum/y:output:0*
T0*
_output_shapes
: i
$MovingAvgQuantize/AssignMinEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o:
-MovingAvgQuantize/AssignMinEma/ReadVariableOpReadVariableOp6movingavgquantize_assignminema_readvariableop_resource*
_output_shapes
: *
dtype0 
"MovingAvgQuantize/AssignMinEma/subSub5MovingAvgQuantize/AssignMinEma/ReadVariableOp:value:0MovingAvgQuantize/Minimum:z:0*
T0*
_output_shapes
: ¡
"MovingAvgQuantize/AssignMinEma/mulMul&MovingAvgQuantize/AssignMinEma/sub:z:0-MovingAvgQuantize/AssignMinEma/decay:output:0*
T0*
_output_shapes
: ü
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOpAssignSubVariableOp6movingavgquantize_assignminema_readvariableop_resource&MovingAvgQuantize/AssignMinEma/mul:z:0.^MovingAvgQuantize/AssignMinEma/ReadVariableOp*
_output_shapes
 *
dtype0i
$MovingAvgQuantize/AssignMaxEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o:
-MovingAvgQuantize/AssignMaxEma/ReadVariableOpReadVariableOp6movingavgquantize_assignmaxema_readvariableop_resource*
_output_shapes
: *
dtype0 
"MovingAvgQuantize/AssignMaxEma/subSub5MovingAvgQuantize/AssignMaxEma/ReadVariableOp:value:0MovingAvgQuantize/Maximum:z:0*
T0*
_output_shapes
: ¡
"MovingAvgQuantize/AssignMaxEma/mulMul&MovingAvgQuantize/AssignMaxEma/sub:z:0-MovingAvgQuantize/AssignMaxEma/decay:output:0*
T0*
_output_shapes
: ü
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpAssignSubVariableOp6movingavgquantize_assignmaxema_readvariableop_resource&MovingAvgQuantize/AssignMaxEma/mul:z:0.^MovingAvgQuantize/AssignMaxEma/ReadVariableOp*
_output_shapes
 *
dtype0Ü
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp6movingavgquantize_assignminema_readvariableop_resource3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp*
_output_shapes
: *
dtype0Þ
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp6movingavgquantize_assignmaxema_readvariableop_resource3^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp*
_output_shapes
: *
dtype0
)MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsRelu:activations:0@MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BMovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@©
1conv2d_2/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
: @*
dtype0
"conv2d_2/kernel/Regularizer/L2LossL2Loss9conv2d_2/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!conv2d_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o; 
conv2d_2/kernel/Regularizer/mulMul*conv2d_2/kernel/Regularizer/mul/x:output:0+conv2d_2/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: 
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¯
NoOpNoOp^BiasAdd/ReadVariableOp^LastValueQuant/AssignMaxLast^LastValueQuant/AssignMinLast'^LastValueQuant/BatchMax/ReadVariableOp'^LastValueQuant/BatchMin/ReadVariableOp@^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpB^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1B^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_23^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp.^MovingAvgQuantize/AssignMaxEma/ReadVariableOp3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp.^MovingAvgQuantize/AssignMinEma/ReadVariableOp9^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12^conv2d_2/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ : : : : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2<
LastValueQuant/AssignMaxLastLastValueQuant/AssignMaxLast2<
LastValueQuant/AssignMinLastLastValueQuant/AssignMinLast2P
&LastValueQuant/BatchMax/ReadVariableOp&LastValueQuant/BatchMax/ReadVariableOp2P
&LastValueQuant/BatchMin/ReadVariableOp&LastValueQuant/BatchMin/ReadVariableOp2
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp2
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_12
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_22h
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2^
-MovingAvgQuantize/AssignMaxEma/ReadVariableOp-MovingAvgQuantize/AssignMaxEma/ReadVariableOp2h
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2^
-MovingAvgQuantize/AssignMinEma/ReadVariableOp-MovingAvgQuantize/AssignMinEma/ReadVariableOp2t
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12f
1conv2d_2/kernel/Regularizer/L2Loss/ReadVariableOp1conv2d_2/kernel/Regularizer/L2Loss/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
å
P
4__inference_quant_max_pooling2d_layer_call_fn_146204

inputs
identityÐ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿKK * 
_read_only_resource_inputs
 *;
config_proto+)

CPU

GPU2*0J
	
   E 8 *X
fSRQ
O__inference_quant_max_pooling2d_layer_call_and_return_conditional_losses_143000h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿKK "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿââ :Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿââ 
 
_user_specified_nameinputs
ú
­
__inference_loss_fn_5_147219K
7dense_kernel_regularizer_l2loss_readvariableop_resource:

identity¢.dense/kernel/Regularizer/L2Loss/ReadVariableOp¨
.dense/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp7dense_kernel_regularizer_l2loss_readvariableop_resource* 
_output_shapes
:
*
dtype0
dense/kernel/Regularizer/L2LossL2Loss6dense/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: c
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0(dense/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: ^
IdentityIdentity dense/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: w
NoOpNoOp/^dense/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2`
.dense/kernel/Regularizer/L2Loss/ReadVariableOp.dense/kernel/Regularizer/L2Loss/ReadVariableOp
°	
¹
__inference_loss_fn_2_147182T
:conv2d_2_kernel_regularizer_l2loss_readvariableop_resource: @
identity¢1conv2d_2/kernel/Regularizer/L2Loss/ReadVariableOp´
1conv2d_2/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp:conv2d_2_kernel_regularizer_l2loss_readvariableop_resource*&
_output_shapes
: @*
dtype0
"conv2d_2/kernel/Regularizer/L2LossL2Loss9conv2d_2/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!conv2d_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o; 
conv2d_2/kernel/Regularizer/mulMul*conv2d_2/kernel/Regularizer/mul/x:output:0+conv2d_2/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: a
IdentityIdentity#conv2d_2/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: z
NoOpNoOp2^conv2d_2/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2f
1conv2d_2/kernel/Regularizer/L2Loss/ReadVariableOp1conv2d_2/kernel/Regularizer/L2Loss/ReadVariableOp
Ë
e
I__inference_quant_flatten_layer_call_and_return_conditional_losses_146745

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ :W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Â
J
.__inference_max_pooling2d_layer_call_fn_147149

inputs
identityå
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *;
config_proto+)

CPU

GPU2*0J
	
   E 8 *R
fMRK
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_142907
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
	

/__inference_quant_conv2d_2_layer_call_fn_146372

inputs!
unknown: @
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3: 
	unknown_4: 
identity¢StatefulPartitionedCall©
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*(
_read_only_resource_inputs

*;
config_proto+)

CPU

GPU2*0J
	
   E 8 *S
fNRL
J__inference_quant_conv2d_2_layer_call_and_return_conditional_losses_143073w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Ë
e
I__inference_quant_flatten_layer_call_and_return_conditional_losses_143178

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ :W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
ø

/__inference_quantize_layer_layer_call_fn_146044

inputs
unknown: 
	unknown_0: 
identity¢StatefulPartitionedCall÷
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿææ*$
_read_only_resource_inputs
*;
config_proto+)

CPU

GPU2*0J
	
   E 8 *S
fNRL
J__inference_quantize_layer_layer_call_and_return_conditional_losses_142950y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿææ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿææ: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿææ
 
_user_specified_nameinputs
Øì
ÁH
!__inference__wrapped_model_142898
conv2d_inpute
[sequential_quantize_layer_allvaluesquantize_fakequantwithminmaxvars_readvariableop_resource: g
]sequential_quantize_layer_allvaluesquantize_fakequantwithminmaxvars_readvariableop_1_resource: z
`sequential_quant_conv2d_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource: p
bsequential_quant_conv2d_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource: p
bsequential_quant_conv2d_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource: E
7sequential_quant_conv2d_biasadd_readvariableop_resource: c
Ysequential_quant_conv2d_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource: e
[sequential_quant_conv2d_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: |
bsequential_quant_conv2d_1_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource:  r
dsequential_quant_conv2d_1_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource: r
dsequential_quant_conv2d_1_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource: G
9sequential_quant_conv2d_1_biasadd_readvariableop_resource: e
[sequential_quant_conv2d_1_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource: g
]sequential_quant_conv2d_1_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: |
bsequential_quant_conv2d_2_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource: @r
dsequential_quant_conv2d_2_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource:@r
dsequential_quant_conv2d_2_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource:@G
9sequential_quant_conv2d_2_biasadd_readvariableop_resource:@e
[sequential_quant_conv2d_2_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource: g
]sequential_quant_conv2d_2_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: |
bsequential_quant_conv2d_3_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource:@@r
dsequential_quant_conv2d_3_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource:@r
dsequential_quant_conv2d_3_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource:@G
9sequential_quant_conv2d_3_biasadd_readvariableop_resource:@e
[sequential_quant_conv2d_3_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource: g
]sequential_quant_conv2d_3_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: |
bsequential_quant_conv2d_4_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource:@ r
dsequential_quant_conv2d_4_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource: r
dsequential_quant_conv2d_4_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource: G
9sequential_quant_conv2d_4_biasadd_readvariableop_resource: e
[sequential_quant_conv2d_4_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource: g
]sequential_quant_conv2d_4_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: i
Usequential_quant_dense_lastvaluequant_fakequantwithminmaxvars_readvariableop_resource:
a
Wsequential_quant_dense_lastvaluequant_fakequantwithminmaxvars_readvariableop_1_resource: a
Wsequential_quant_dense_lastvaluequant_fakequantwithminmaxvars_readvariableop_2_resource: E
6sequential_quant_dense_biasadd_readvariableop_resource:	b
Xsequential_quant_dense_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource: d
Zsequential_quant_dense_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: k
Wsequential_quant_dense_1_lastvaluequant_fakequantwithminmaxvars_readvariableop_resource:
c
Ysequential_quant_dense_1_lastvaluequant_fakequantwithminmaxvars_readvariableop_1_resource: c
Ysequential_quant_dense_1_lastvaluequant_fakequantwithminmaxvars_readvariableop_2_resource: G
8sequential_quant_dense_1_biasadd_readvariableop_resource:	d
Zsequential_quant_dense_1_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource: f
\sequential_quant_dense_1_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: j
Wsequential_quant_dense_2_lastvaluequant_fakequantwithminmaxvars_readvariableop_resource:	c
Ysequential_quant_dense_2_lastvaluequant_fakequantwithminmaxvars_readvariableop_1_resource: c
Ysequential_quant_dense_2_lastvaluequant_fakequantwithminmaxvars_readvariableop_2_resource: F
8sequential_quant_dense_2_biasadd_readvariableop_resource:d
Zsequential_quant_dense_2_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource: f
\sequential_quant_dense_2_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: 
identity¢.sequential/quant_conv2d/BiasAdd/ReadVariableOp¢Wsequential/quant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp¢Ysequential/quant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1¢Ysequential/quant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2¢Psequential/quant_conv2d/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp¢Rsequential/quant_conv2d/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1¢0sequential/quant_conv2d_1/BiasAdd/ReadVariableOp¢Ysequential/quant_conv2d_1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp¢[sequential/quant_conv2d_1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1¢[sequential/quant_conv2d_1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2¢Rsequential/quant_conv2d_1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp¢Tsequential/quant_conv2d_1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1¢0sequential/quant_conv2d_2/BiasAdd/ReadVariableOp¢Ysequential/quant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp¢[sequential/quant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1¢[sequential/quant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2¢Rsequential/quant_conv2d_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp¢Tsequential/quant_conv2d_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1¢0sequential/quant_conv2d_3/BiasAdd/ReadVariableOp¢Ysequential/quant_conv2d_3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp¢[sequential/quant_conv2d_3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1¢[sequential/quant_conv2d_3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2¢Rsequential/quant_conv2d_3/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp¢Tsequential/quant_conv2d_3/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1¢0sequential/quant_conv2d_4/BiasAdd/ReadVariableOp¢Ysequential/quant_conv2d_4/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp¢[sequential/quant_conv2d_4/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1¢[sequential/quant_conv2d_4/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2¢Rsequential/quant_conv2d_4/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp¢Tsequential/quant_conv2d_4/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1¢-sequential/quant_dense/BiasAdd/ReadVariableOp¢Lsequential/quant_dense/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp¢Nsequential/quant_dense/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1¢Nsequential/quant_dense/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2¢Osequential/quant_dense/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp¢Qsequential/quant_dense/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1¢/sequential/quant_dense_1/BiasAdd/ReadVariableOp¢Nsequential/quant_dense_1/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp¢Psequential/quant_dense_1/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1¢Psequential/quant_dense_1/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2¢Qsequential/quant_dense_1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp¢Ssequential/quant_dense_1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1¢/sequential/quant_dense_2/BiasAdd/ReadVariableOp¢Nsequential/quant_dense_2/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp¢Psequential/quant_dense_2/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1¢Psequential/quant_dense_2/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2¢Qsequential/quant_dense_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp¢Ssequential/quant_dense_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1¢Rsequential/quantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp¢Tsequential/quantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1æ
Rsequential/quantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp[sequential_quantize_layer_allvaluesquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype0ê
Tsequential/quantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp]sequential_quantize_layer_allvaluesquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0Ù
Csequential/quantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsconv2d_inputZsequential/quantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0\sequential/quantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿææ
Wsequential/quant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpReadVariableOp`sequential_quant_conv2d_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource*&
_output_shapes
: *
dtype0ø
Ysequential/quant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ReadVariableOpbsequential_quant_conv2d_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource*
_output_shapes
: *
dtype0ø
Ysequential/quant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ReadVariableOpbsequential_quant_conv2d_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource*
_output_shapes
: *
dtype0Ð
Hsequential/quant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel!FakeQuantWithMinMaxVarsPerChannel_sequential/quant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp:value:0asequential/quant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1:value:0asequential/quant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2:value:0*&
_output_shapes
: *
narrow_range(°
sequential/quant_conv2d/Conv2DConv2DMsequential/quantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars:outputs:0Rsequential/quant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel:outputs:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿââ *
paddingVALID*
strides
¢
.sequential/quant_conv2d/BiasAdd/ReadVariableOpReadVariableOp7sequential_quant_conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0Ç
sequential/quant_conv2d/BiasAddBiasAdd'sequential/quant_conv2d/Conv2D:output:06sequential/quant_conv2d/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿââ 
sequential/quant_conv2d/ReluRelu(sequential/quant_conv2d/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿââ â
Psequential/quant_conv2d/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpYsequential_quant_conv2d_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype0æ
Rsequential/quant_conv2d/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp[sequential_quant_conv2d_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0ñ
Asequential/quant_conv2d/MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars*sequential/quant_conv2d/Relu:activations:0Xsequential/quant_conv2d/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Zsequential/quant_conv2d/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿââ ë
&sequential/quant_max_pooling2d/MaxPoolMaxPoolKsequential/quant_conv2d/MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿKK *
ksize
*
paddingVALID*
strides

Ysequential/quant_conv2d_1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpReadVariableOpbsequential_quant_conv2d_1_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource*&
_output_shapes
:  *
dtype0ü
[sequential/quant_conv2d_1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ReadVariableOpdsequential_quant_conv2d_1_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource*
_output_shapes
: *
dtype0ü
[sequential/quant_conv2d_1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ReadVariableOpdsequential_quant_conv2d_1_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource*
_output_shapes
: *
dtype0Ø
Jsequential/quant_conv2d_1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel!FakeQuantWithMinMaxVarsPerChannelasequential/quant_conv2d_1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp:value:0csequential/quant_conv2d_1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1:value:0csequential/quant_conv2d_1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2:value:0*&
_output_shapes
:  *
narrow_range(
 sequential/quant_conv2d_1/Conv2DConv2D/sequential/quant_max_pooling2d/MaxPool:output:0Tsequential/quant_conv2d_1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel:outputs:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿKK *
paddingSAME*
strides
¦
0sequential/quant_conv2d_1/BiasAdd/ReadVariableOpReadVariableOp9sequential_quant_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0Ë
!sequential/quant_conv2d_1/BiasAddBiasAdd)sequential/quant_conv2d_1/Conv2D:output:08sequential/quant_conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿKK 
sequential/quant_conv2d_1/ReluRelu*sequential/quant_conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿKK æ
Rsequential/quant_conv2d_1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp[sequential_quant_conv2d_1_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype0ê
Tsequential/quant_conv2d_1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp]sequential_quant_conv2d_1_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0÷
Csequential/quant_conv2d_1/MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars,sequential/quant_conv2d_1/Relu:activations:0Zsequential/quant_conv2d_1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0\sequential/quant_conv2d_1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿKK ï
(sequential/quant_max_pooling2d_1/MaxPoolMaxPoolMsequential/quant_conv2d_1/MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
ksize
*
paddingVALID*
strides

Ysequential/quant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpReadVariableOpbsequential_quant_conv2d_2_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource*&
_output_shapes
: @*
dtype0ü
[sequential/quant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ReadVariableOpdsequential_quant_conv2d_2_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource*
_output_shapes
:@*
dtype0ü
[sequential/quant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ReadVariableOpdsequential_quant_conv2d_2_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource*
_output_shapes
:@*
dtype0Ø
Jsequential/quant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel!FakeQuantWithMinMaxVarsPerChannelasequential/quant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp:value:0csequential/quant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1:value:0csequential/quant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2:value:0*&
_output_shapes
: @*
narrow_range(
 sequential/quant_conv2d_2/Conv2DConv2D1sequential/quant_max_pooling2d_1/MaxPool:output:0Tsequential/quant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel:outputs:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingSAME*
strides
¦
0sequential/quant_conv2d_2/BiasAdd/ReadVariableOpReadVariableOp9sequential_quant_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ë
!sequential/quant_conv2d_2/BiasAddBiasAdd)sequential/quant_conv2d_2/Conv2D:output:08sequential/quant_conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
sequential/quant_conv2d_2/ReluRelu*sequential/quant_conv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@æ
Rsequential/quant_conv2d_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp[sequential_quant_conv2d_2_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype0ê
Tsequential/quant_conv2d_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp]sequential_quant_conv2d_2_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0÷
Csequential/quant_conv2d_2/MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars,sequential/quant_conv2d_2/Relu:activations:0Zsequential/quant_conv2d_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0\sequential/quant_conv2d_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
Ysequential/quant_conv2d_3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpReadVariableOpbsequential_quant_conv2d_3_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource*&
_output_shapes
:@@*
dtype0ü
[sequential/quant_conv2d_3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ReadVariableOpdsequential_quant_conv2d_3_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource*
_output_shapes
:@*
dtype0ü
[sequential/quant_conv2d_3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ReadVariableOpdsequential_quant_conv2d_3_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource*
_output_shapes
:@*
dtype0Ø
Jsequential/quant_conv2d_3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel!FakeQuantWithMinMaxVarsPerChannelasequential/quant_conv2d_3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp:value:0csequential/quant_conv2d_3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1:value:0csequential/quant_conv2d_3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2:value:0*&
_output_shapes
:@@*
narrow_range(±
 sequential/quant_conv2d_3/Conv2DConv2DMsequential/quant_conv2d_2/MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0Tsequential/quant_conv2d_3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel:outputs:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingSAME*
strides
¦
0sequential/quant_conv2d_3/BiasAdd/ReadVariableOpReadVariableOp9sequential_quant_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ë
!sequential/quant_conv2d_3/BiasAddBiasAdd)sequential/quant_conv2d_3/Conv2D:output:08sequential/quant_conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
sequential/quant_conv2d_3/ReluRelu*sequential/quant_conv2d_3/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@æ
Rsequential/quant_conv2d_3/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp[sequential_quant_conv2d_3_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype0ê
Tsequential/quant_conv2d_3/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp]sequential_quant_conv2d_3_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0÷
Csequential/quant_conv2d_3/MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars,sequential/quant_conv2d_3/Relu:activations:0Zsequential/quant_conv2d_3/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0\sequential/quant_conv2d_3/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
Ysequential/quant_conv2d_4/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpReadVariableOpbsequential_quant_conv2d_4_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource*&
_output_shapes
:@ *
dtype0ü
[sequential/quant_conv2d_4/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ReadVariableOpdsequential_quant_conv2d_4_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource*
_output_shapes
: *
dtype0ü
[sequential/quant_conv2d_4/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ReadVariableOpdsequential_quant_conv2d_4_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource*
_output_shapes
: *
dtype0Ø
Jsequential/quant_conv2d_4/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel!FakeQuantWithMinMaxVarsPerChannelasequential/quant_conv2d_4/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp:value:0csequential/quant_conv2d_4/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1:value:0csequential/quant_conv2d_4/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2:value:0*&
_output_shapes
:@ *
narrow_range(±
 sequential/quant_conv2d_4/Conv2DConv2DMsequential/quant_conv2d_3/MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0Tsequential/quant_conv2d_4/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel:outputs:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
¦
0sequential/quant_conv2d_4/BiasAdd/ReadVariableOpReadVariableOp9sequential_quant_conv2d_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0Ë
!sequential/quant_conv2d_4/BiasAddBiasAdd)sequential/quant_conv2d_4/Conv2D:output:08sequential/quant_conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
sequential/quant_conv2d_4/ReluRelu*sequential/quant_conv2d_4/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ æ
Rsequential/quant_conv2d_4/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp[sequential_quant_conv2d_4_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype0ê
Tsequential/quant_conv2d_4/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp]sequential_quant_conv2d_4_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0÷
Csequential/quant_conv2d_4/MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars,sequential/quant_conv2d_4/Relu:activations:0Zsequential/quant_conv2d_4/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0\sequential/quant_conv2d_4/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ï
(sequential/quant_max_pooling2d_2/MaxPoolMaxPoolMsequential/quant_conv2d_4/MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
ksize
*
paddingVALID*
strides
o
sequential/quant_flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   º
 sequential/quant_flatten/ReshapeReshape1sequential/quant_max_pooling2d_2/MaxPool:output:0'sequential/quant_flatten/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿä
Lsequential/quant_dense/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpUsequential_quant_dense_lastvaluequant_fakequantwithminmaxvars_readvariableop_resource* 
_output_shapes
:
*
dtype0Þ
Nsequential/quant_dense/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpWsequential_quant_dense_lastvaluequant_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0Þ
Nsequential/quant_dense/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2ReadVariableOpWsequential_quant_dense_lastvaluequant_fakequantwithminmaxvars_readvariableop_2_resource*
_output_shapes
: *
dtype0
=sequential/quant_dense/LastValueQuant/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsTsequential/quant_dense/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Vsequential/quant_dense/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0Vsequential/quant_dense/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2:value:0* 
_output_shapes
:
*
narrow_range(Î
sequential/quant_dense/MatMulMatMul)sequential/quant_flatten/Reshape:output:0Gsequential/quant_dense/LastValueQuant/FakeQuantWithMinMaxVars:outputs:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¡
-sequential/quant_dense/BiasAdd/ReadVariableOpReadVariableOp6sequential_quant_dense_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0¼
sequential/quant_dense/BiasAddBiasAdd'sequential/quant_dense/MatMul:product:05sequential/quant_dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
sequential/quant_dense/ReluRelu'sequential/quant_dense/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà
Osequential/quant_dense/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpXsequential_quant_dense_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype0ä
Qsequential/quant_dense/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpZsequential_quant_dense_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0ä
@sequential/quant_dense/MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars)sequential/quant_dense/Relu:activations:0Wsequential/quant_dense/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Ysequential/quant_dense/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
!sequential/quant_dropout/IdentityIdentityJsequential/quant_dense/MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè
Nsequential/quant_dense_1/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpWsequential_quant_dense_1_lastvaluequant_fakequantwithminmaxvars_readvariableop_resource* 
_output_shapes
:
*
dtype0â
Psequential/quant_dense_1/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpYsequential_quant_dense_1_lastvaluequant_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0â
Psequential/quant_dense_1/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2ReadVariableOpYsequential_quant_dense_1_lastvaluequant_fakequantwithminmaxvars_readvariableop_2_resource*
_output_shapes
: *
dtype0
?sequential/quant_dense_1/LastValueQuant/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsVsequential/quant_dense_1/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Xsequential/quant_dense_1/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0Xsequential/quant_dense_1/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2:value:0* 
_output_shapes
:
*
narrow_range(Ó
sequential/quant_dense_1/MatMulMatMul*sequential/quant_dropout/Identity:output:0Isequential/quant_dense_1/LastValueQuant/FakeQuantWithMinMaxVars:outputs:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥
/sequential/quant_dense_1/BiasAdd/ReadVariableOpReadVariableOp8sequential_quant_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Â
 sequential/quant_dense_1/BiasAddBiasAdd)sequential/quant_dense_1/MatMul:product:07sequential/quant_dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
sequential/quant_dense_1/ReluRelu)sequential/quant_dense_1/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿä
Qsequential/quant_dense_1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpZsequential_quant_dense_1_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype0è
Ssequential/quant_dense_1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp\sequential_quant_dense_1_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0ì
Bsequential/quant_dense_1/MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars+sequential/quant_dense_1/Relu:activations:0Ysequential/quant_dense_1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0[sequential/quant_dense_1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°
#sequential/quant_dropout_1/IdentityIdentityLsequential/quant_dense_1/MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç
Nsequential/quant_dense_2/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpWsequential_quant_dense_2_lastvaluequant_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
:	*
dtype0â
Psequential/quant_dense_2/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpYsequential_quant_dense_2_lastvaluequant_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0â
Psequential/quant_dense_2/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2ReadVariableOpYsequential_quant_dense_2_lastvaluequant_fakequantwithminmaxvars_readvariableop_2_resource*
_output_shapes
: *
dtype0
?sequential/quant_dense_2/LastValueQuant/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsVsequential/quant_dense_2/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Xsequential/quant_dense_2/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0Xsequential/quant_dense_2/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2:value:0*
_output_shapes
:	*
narrow_range(Ô
sequential/quant_dense_2/MatMulMatMul,sequential/quant_dropout_1/Identity:output:0Isequential/quant_dense_2/LastValueQuant/FakeQuantWithMinMaxVars:outputs:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤
/sequential/quant_dense_2/BiasAdd/ReadVariableOpReadVariableOp8sequential_quant_dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Á
 sequential/quant_dense_2/BiasAddBiasAdd)sequential/quant_dense_2/MatMul:product:07sequential/quant_dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿä
Qsequential/quant_dense_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpZsequential_quant_dense_2_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype0è
Ssequential/quant_dense_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp\sequential_quant_dense_2_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0é
Bsequential/quant_dense_2/MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars)sequential/quant_dense_2/BiasAdd:output:0Ysequential/quant_dense_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0[sequential/quant_dense_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ«
 sequential/quant_dense_2/SoftmaxSoftmaxLsequential/quant_dense_2/MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿy
IdentityIdentity*sequential/quant_dense_2/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¦ 
NoOpNoOp/^sequential/quant_conv2d/BiasAdd/ReadVariableOpX^sequential/quant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpZ^sequential/quant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1Z^sequential/quant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2Q^sequential/quant_conv2d/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpS^sequential/quant_conv2d/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_11^sequential/quant_conv2d_1/BiasAdd/ReadVariableOpZ^sequential/quant_conv2d_1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp\^sequential/quant_conv2d_1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1\^sequential/quant_conv2d_1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2S^sequential/quant_conv2d_1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpU^sequential/quant_conv2d_1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_11^sequential/quant_conv2d_2/BiasAdd/ReadVariableOpZ^sequential/quant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp\^sequential/quant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1\^sequential/quant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2S^sequential/quant_conv2d_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpU^sequential/quant_conv2d_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_11^sequential/quant_conv2d_3/BiasAdd/ReadVariableOpZ^sequential/quant_conv2d_3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp\^sequential/quant_conv2d_3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1\^sequential/quant_conv2d_3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2S^sequential/quant_conv2d_3/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpU^sequential/quant_conv2d_3/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_11^sequential/quant_conv2d_4/BiasAdd/ReadVariableOpZ^sequential/quant_conv2d_4/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp\^sequential/quant_conv2d_4/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1\^sequential/quant_conv2d_4/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2S^sequential/quant_conv2d_4/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpU^sequential/quant_conv2d_4/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1.^sequential/quant_dense/BiasAdd/ReadVariableOpM^sequential/quant_dense/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpO^sequential/quant_dense/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1O^sequential/quant_dense/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2P^sequential/quant_dense/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpR^sequential/quant_dense/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_10^sequential/quant_dense_1/BiasAdd/ReadVariableOpO^sequential/quant_dense_1/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpQ^sequential/quant_dense_1/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1Q^sequential/quant_dense_1/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2R^sequential/quant_dense_1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpT^sequential/quant_dense_1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_10^sequential/quant_dense_2/BiasAdd/ReadVariableOpO^sequential/quant_dense_2/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpQ^sequential/quant_dense_2/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1Q^sequential/quant_dense_2/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2R^sequential/quant_dense_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpT^sequential/quant_dense_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1S^sequential/quantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOpU^sequential/quantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:ÿÿÿÿÿÿÿÿÿææ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2`
.sequential/quant_conv2d/BiasAdd/ReadVariableOp.sequential/quant_conv2d/BiasAdd/ReadVariableOp2²
Wsequential/quant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpWsequential/quant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp2¶
Ysequential/quant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1Ysequential/quant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_12¶
Ysequential/quant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2Ysequential/quant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_22¤
Psequential/quant_conv2d/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpPsequential/quant_conv2d/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2¨
Rsequential/quant_conv2d/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Rsequential/quant_conv2d/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12d
0sequential/quant_conv2d_1/BiasAdd/ReadVariableOp0sequential/quant_conv2d_1/BiasAdd/ReadVariableOp2¶
Ysequential/quant_conv2d_1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpYsequential/quant_conv2d_1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp2º
[sequential/quant_conv2d_1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1[sequential/quant_conv2d_1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_12º
[sequential/quant_conv2d_1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2[sequential/quant_conv2d_1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_22¨
Rsequential/quant_conv2d_1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpRsequential/quant_conv2d_1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2¬
Tsequential/quant_conv2d_1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Tsequential/quant_conv2d_1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12d
0sequential/quant_conv2d_2/BiasAdd/ReadVariableOp0sequential/quant_conv2d_2/BiasAdd/ReadVariableOp2¶
Ysequential/quant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpYsequential/quant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp2º
[sequential/quant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1[sequential/quant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_12º
[sequential/quant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2[sequential/quant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_22¨
Rsequential/quant_conv2d_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpRsequential/quant_conv2d_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2¬
Tsequential/quant_conv2d_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Tsequential/quant_conv2d_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12d
0sequential/quant_conv2d_3/BiasAdd/ReadVariableOp0sequential/quant_conv2d_3/BiasAdd/ReadVariableOp2¶
Ysequential/quant_conv2d_3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpYsequential/quant_conv2d_3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp2º
[sequential/quant_conv2d_3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1[sequential/quant_conv2d_3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_12º
[sequential/quant_conv2d_3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2[sequential/quant_conv2d_3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_22¨
Rsequential/quant_conv2d_3/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpRsequential/quant_conv2d_3/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2¬
Tsequential/quant_conv2d_3/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Tsequential/quant_conv2d_3/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12d
0sequential/quant_conv2d_4/BiasAdd/ReadVariableOp0sequential/quant_conv2d_4/BiasAdd/ReadVariableOp2¶
Ysequential/quant_conv2d_4/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpYsequential/quant_conv2d_4/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp2º
[sequential/quant_conv2d_4/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1[sequential/quant_conv2d_4/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_12º
[sequential/quant_conv2d_4/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2[sequential/quant_conv2d_4/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_22¨
Rsequential/quant_conv2d_4/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpRsequential/quant_conv2d_4/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2¬
Tsequential/quant_conv2d_4/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Tsequential/quant_conv2d_4/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12^
-sequential/quant_dense/BiasAdd/ReadVariableOp-sequential/quant_dense/BiasAdd/ReadVariableOp2
Lsequential/quant_dense/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpLsequential/quant_dense/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp2 
Nsequential/quant_dense/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1Nsequential/quant_dense/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_12 
Nsequential/quant_dense/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2Nsequential/quant_dense/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_22¢
Osequential/quant_dense/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpOsequential/quant_dense/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2¦
Qsequential/quant_dense/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Qsequential/quant_dense/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12b
/sequential/quant_dense_1/BiasAdd/ReadVariableOp/sequential/quant_dense_1/BiasAdd/ReadVariableOp2 
Nsequential/quant_dense_1/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpNsequential/quant_dense_1/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp2¤
Psequential/quant_dense_1/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1Psequential/quant_dense_1/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_12¤
Psequential/quant_dense_1/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2Psequential/quant_dense_1/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_22¦
Qsequential/quant_dense_1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpQsequential/quant_dense_1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2ª
Ssequential/quant_dense_1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Ssequential/quant_dense_1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12b
/sequential/quant_dense_2/BiasAdd/ReadVariableOp/sequential/quant_dense_2/BiasAdd/ReadVariableOp2 
Nsequential/quant_dense_2/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpNsequential/quant_dense_2/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp2¤
Psequential/quant_dense_2/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1Psequential/quant_dense_2/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_12¤
Psequential/quant_dense_2/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2Psequential/quant_dense_2/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_22¦
Qsequential/quant_dense_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpQsequential/quant_dense_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2ª
Ssequential/quant_dense_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Ssequential/quant_dense_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12¨
Rsequential/quantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOpRsequential/quantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2¬
Tsequential/quantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Tsequential/quantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:_ [
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿææ
&
_user_specified_nameconv2d_input
¤X
¶	
H__inference_quant_conv2d_layer_call_and_return_conditional_losses_144235

inputsI
/lastvaluequant_batchmin_readvariableop_resource: 3
%lastvaluequant_assignminlast_resource: 3
%lastvaluequant_assignmaxlast_resource: -
biasadd_readvariableop_resource: @
6movingavgquantize_assignminema_readvariableop_resource: @
6movingavgquantize_assignmaxema_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢LastValueQuant/AssignMaxLast¢LastValueQuant/AssignMinLast¢&LastValueQuant/BatchMax/ReadVariableOp¢&LastValueQuant/BatchMin/ReadVariableOp¢?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp¢ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1¢ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2¢2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp¢-MovingAvgQuantize/AssignMaxEma/ReadVariableOp¢2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp¢-MovingAvgQuantize/AssignMinEma/ReadVariableOp¢8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp¢:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1¢/conv2d/kernel/Regularizer/L2Loss/ReadVariableOp
&LastValueQuant/BatchMin/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
: *
dtype0~
)LastValueQuant/BatchMin/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          §
LastValueQuant/BatchMinMin.LastValueQuant/BatchMin/ReadVariableOp:value:02LastValueQuant/BatchMin/reduction_indices:output:0*
T0*
_output_shapes
: 
&LastValueQuant/BatchMax/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
: *
dtype0~
)LastValueQuant/BatchMax/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          §
LastValueQuant/BatchMaxMax.LastValueQuant/BatchMax/ReadVariableOp:value:02LastValueQuant/BatchMax/reduction_indices:output:0*
T0*
_output_shapes
: ]
LastValueQuant/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ¿
LastValueQuant/truedivRealDiv LastValueQuant/BatchMax:output:0!LastValueQuant/truediv/y:output:0*
T0*
_output_shapes
: 
LastValueQuant/MinimumMinimum LastValueQuant/BatchMin:output:0LastValueQuant/truediv:z:0*
T0*
_output_shapes
: Y
LastValueQuant/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ¿
LastValueQuant/mulMul LastValueQuant/BatchMin:output:0LastValueQuant/mul/y:output:0*
T0*
_output_shapes
: 
LastValueQuant/MaximumMaximum LastValueQuant/BatchMax:output:0LastValueQuant/mul:z:0*
T0*
_output_shapes
: ¬
LastValueQuant/AssignMinLastAssignVariableOp%lastvaluequant_assignminlast_resourceLastValueQuant/Minimum:z:0*
_output_shapes
 *
dtype0*
validate_shape(¬
LastValueQuant/AssignMaxLastAssignVariableOp%lastvaluequant_assignmaxlast_resourceLastValueQuant/Maximum:z:0*
_output_shapes
 *
dtype0*
validate_shape(·
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
: *
dtype0Â
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ReadVariableOp%lastvaluequant_assignminlast_resource^LastValueQuant/AssignMinLast*
_output_shapes
: *
dtype0Â
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ReadVariableOp%lastvaluequant_assignmaxlast_resource^LastValueQuant/AssignMaxLast*
_output_shapes
: *
dtype0ð
0LastValueQuant/FakeQuantWithMinMaxVarsPerChannel!FakeQuantWithMinMaxVarsPerChannelGLastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2:value:0*&
_output_shapes
: *
narrow_range(¹
Conv2DConv2Dinputs:LastValueQuant/FakeQuantWithMinMaxVarsPerChannel:outputs:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿââ *
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿââ Z
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿââ p
MovingAvgQuantize/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             x
MovingAvgQuantize/BatchMinMinRelu:activations:0 MovingAvgQuantize/Const:output:0*
T0*
_output_shapes
: r
MovingAvgQuantize/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             z
MovingAvgQuantize/BatchMaxMaxRelu:activations:0"MovingAvgQuantize/Const_1:output:0*
T0*
_output_shapes
: `
MovingAvgQuantize/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
MovingAvgQuantize/MinimumMinimum#MovingAvgQuantize/BatchMin:output:0$MovingAvgQuantize/Minimum/y:output:0*
T0*
_output_shapes
: `
MovingAvgQuantize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
MovingAvgQuantize/MaximumMaximum#MovingAvgQuantize/BatchMax:output:0$MovingAvgQuantize/Maximum/y:output:0*
T0*
_output_shapes
: i
$MovingAvgQuantize/AssignMinEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o:
-MovingAvgQuantize/AssignMinEma/ReadVariableOpReadVariableOp6movingavgquantize_assignminema_readvariableop_resource*
_output_shapes
: *
dtype0 
"MovingAvgQuantize/AssignMinEma/subSub5MovingAvgQuantize/AssignMinEma/ReadVariableOp:value:0MovingAvgQuantize/Minimum:z:0*
T0*
_output_shapes
: ¡
"MovingAvgQuantize/AssignMinEma/mulMul&MovingAvgQuantize/AssignMinEma/sub:z:0-MovingAvgQuantize/AssignMinEma/decay:output:0*
T0*
_output_shapes
: ü
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOpAssignSubVariableOp6movingavgquantize_assignminema_readvariableop_resource&MovingAvgQuantize/AssignMinEma/mul:z:0.^MovingAvgQuantize/AssignMinEma/ReadVariableOp*
_output_shapes
 *
dtype0i
$MovingAvgQuantize/AssignMaxEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o:
-MovingAvgQuantize/AssignMaxEma/ReadVariableOpReadVariableOp6movingavgquantize_assignmaxema_readvariableop_resource*
_output_shapes
: *
dtype0 
"MovingAvgQuantize/AssignMaxEma/subSub5MovingAvgQuantize/AssignMaxEma/ReadVariableOp:value:0MovingAvgQuantize/Maximum:z:0*
T0*
_output_shapes
: ¡
"MovingAvgQuantize/AssignMaxEma/mulMul&MovingAvgQuantize/AssignMaxEma/sub:z:0-MovingAvgQuantize/AssignMaxEma/decay:output:0*
T0*
_output_shapes
: ü
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpAssignSubVariableOp6movingavgquantize_assignmaxema_readvariableop_resource&MovingAvgQuantize/AssignMaxEma/mul:z:0.^MovingAvgQuantize/AssignMaxEma/ReadVariableOp*
_output_shapes
 *
dtype0Ü
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp6movingavgquantize_assignminema_readvariableop_resource3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp*
_output_shapes
: *
dtype0Þ
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp6movingavgquantize_assignmaxema_readvariableop_resource3^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp*
_output_shapes
: *
dtype0
)MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsRelu:activations:0@MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BMovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿââ §
/conv2d/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
: *
dtype0
 conv2d/kernel/Regularizer/L2LossL2Loss7conv2d/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: d
conv2d/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;
conv2d/kernel/Regularizer/mulMul(conv2d/kernel/Regularizer/mul/x:output:0)conv2d/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: 
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿââ ­
NoOpNoOp^BiasAdd/ReadVariableOp^LastValueQuant/AssignMaxLast^LastValueQuant/AssignMinLast'^LastValueQuant/BatchMax/ReadVariableOp'^LastValueQuant/BatchMin/ReadVariableOp@^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpB^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1B^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_23^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp.^MovingAvgQuantize/AssignMaxEma/ReadVariableOp3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp.^MovingAvgQuantize/AssignMinEma/ReadVariableOp9^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_10^conv2d/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):ÿÿÿÿÿÿÿÿÿææ: : : : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2<
LastValueQuant/AssignMaxLastLastValueQuant/AssignMaxLast2<
LastValueQuant/AssignMinLastLastValueQuant/AssignMinLast2P
&LastValueQuant/BatchMax/ReadVariableOp&LastValueQuant/BatchMax/ReadVariableOp2P
&LastValueQuant/BatchMin/ReadVariableOp&LastValueQuant/BatchMin/ReadVariableOp2
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp2
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_12
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_22h
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2^
-MovingAvgQuantize/AssignMaxEma/ReadVariableOp-MovingAvgQuantize/AssignMaxEma/ReadVariableOp2h
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2^
-MovingAvgQuantize/AssignMinEma/ReadVariableOp-MovingAvgQuantize/AssignMinEma/ReadVariableOp2t
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12b
/conv2d/kernel/Regularizer/L2Loss/ReadVariableOp/conv2d/kernel/Regularizer/L2Loss/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿææ
 
_user_specified_nameinputs
Þ~
´
F__inference_sequential_layer_call_and_return_conditional_losses_143336

inputs
quantize_layer_142951: 
quantize_layer_142953: -
quant_conv2d_142982: !
quant_conv2d_142984: !
quant_conv2d_142986: !
quant_conv2d_142988: 
quant_conv2d_142990: 
quant_conv2d_142992: /
quant_conv2d_1_143028:  #
quant_conv2d_1_143030: #
quant_conv2d_1_143032: #
quant_conv2d_1_143034: 
quant_conv2d_1_143036: 
quant_conv2d_1_143038: /
quant_conv2d_2_143074: @#
quant_conv2d_2_143076:@#
quant_conv2d_2_143078:@#
quant_conv2d_2_143080:@
quant_conv2d_2_143082: 
quant_conv2d_2_143084: /
quant_conv2d_3_143113:@@#
quant_conv2d_3_143115:@#
quant_conv2d_3_143117:@#
quant_conv2d_3_143119:@
quant_conv2d_3_143121: 
quant_conv2d_3_143123: /
quant_conv2d_4_143152:@ #
quant_conv2d_4_143154: #
quant_conv2d_4_143156: #
quant_conv2d_4_143158: 
quant_conv2d_4_143160: 
quant_conv2d_4_143162: &
quant_dense_143206:

quant_dense_143208: 
quant_dense_143210: !
quant_dense_143212:	
quant_dense_143214: 
quant_dense_143216: (
quant_dense_1_143252:

quant_dense_1_143254: 
quant_dense_1_143256: #
quant_dense_1_143258:	
quant_dense_1_143260: 
quant_dense_1_143262: '
quant_dense_2_143294:	
quant_dense_2_143296: 
quant_dense_2_143298: "
quant_dense_2_143300:
quant_dense_2_143302: 
quant_dense_2_143304: 
identity¢/conv2d/kernel/Regularizer/L2Loss/ReadVariableOp¢1conv2d_1/kernel/Regularizer/L2Loss/ReadVariableOp¢1conv2d_2/kernel/Regularizer/L2Loss/ReadVariableOp¢1conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOp¢1conv2d_4/kernel/Regularizer/L2Loss/ReadVariableOp¢.dense/kernel/Regularizer/L2Loss/ReadVariableOp¢0dense_1/kernel/Regularizer/L2Loss/ReadVariableOp¢$quant_conv2d/StatefulPartitionedCall¢&quant_conv2d_1/StatefulPartitionedCall¢&quant_conv2d_2/StatefulPartitionedCall¢&quant_conv2d_3/StatefulPartitionedCall¢&quant_conv2d_4/StatefulPartitionedCall¢#quant_dense/StatefulPartitionedCall¢%quant_dense_1/StatefulPartitionedCall¢%quant_dense_2/StatefulPartitionedCall¢&quantize_layer/StatefulPartitionedCall 
&quantize_layer/StatefulPartitionedCallStatefulPartitionedCallinputsquantize_layer_142951quantize_layer_142953*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿææ*$
_read_only_resource_inputs
*;
config_proto+)

CPU

GPU2*0J
	
   E 8 *S
fNRL
J__inference_quantize_layer_layer_call_and_return_conditional_losses_142950
$quant_conv2d/StatefulPartitionedCallStatefulPartitionedCall/quantize_layer/StatefulPartitionedCall:output:0quant_conv2d_142982quant_conv2d_142984quant_conv2d_142986quant_conv2d_142988quant_conv2d_142990quant_conv2d_142992*
Tin
	2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿââ *(
_read_only_resource_inputs

*;
config_proto+)

CPU

GPU2*0J
	
   E 8 *Q
fLRJ
H__inference_quant_conv2d_layer_call_and_return_conditional_losses_142981
#quant_max_pooling2d/PartitionedCallPartitionedCall-quant_conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿKK * 
_read_only_resource_inputs
 *;
config_proto+)

CPU

GPU2*0J
	
   E 8 *X
fSRQ
O__inference_quant_max_pooling2d_layer_call_and_return_conditional_losses_143000¨
&quant_conv2d_1/StatefulPartitionedCallStatefulPartitionedCall,quant_max_pooling2d/PartitionedCall:output:0quant_conv2d_1_143028quant_conv2d_1_143030quant_conv2d_1_143032quant_conv2d_1_143034quant_conv2d_1_143036quant_conv2d_1_143038*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿKK *(
_read_only_resource_inputs

*;
config_proto+)

CPU

GPU2*0J
	
   E 8 *S
fNRL
J__inference_quant_conv2d_1_layer_call_and_return_conditional_losses_143027
%quant_max_pooling2d_1/PartitionedCallPartitionedCall/quant_conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *;
config_proto+)

CPU

GPU2*0J
	
   E 8 *Z
fURS
Q__inference_quant_max_pooling2d_1_layer_call_and_return_conditional_losses_143046ª
&quant_conv2d_2/StatefulPartitionedCallStatefulPartitionedCall.quant_max_pooling2d_1/PartitionedCall:output:0quant_conv2d_2_143074quant_conv2d_2_143076quant_conv2d_2_143078quant_conv2d_2_143080quant_conv2d_2_143082quant_conv2d_2_143084*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*(
_read_only_resource_inputs

*;
config_proto+)

CPU

GPU2*0J
	
   E 8 *S
fNRL
J__inference_quant_conv2d_2_layer_call_and_return_conditional_losses_143073«
&quant_conv2d_3/StatefulPartitionedCallStatefulPartitionedCall/quant_conv2d_2/StatefulPartitionedCall:output:0quant_conv2d_3_143113quant_conv2d_3_143115quant_conv2d_3_143117quant_conv2d_3_143119quant_conv2d_3_143121quant_conv2d_3_143123*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*(
_read_only_resource_inputs

*;
config_proto+)

CPU

GPU2*0J
	
   E 8 *S
fNRL
J__inference_quant_conv2d_3_layer_call_and_return_conditional_losses_143112«
&quant_conv2d_4/StatefulPartitionedCallStatefulPartitionedCall/quant_conv2d_3/StatefulPartitionedCall:output:0quant_conv2d_4_143152quant_conv2d_4_143154quant_conv2d_4_143156quant_conv2d_4_143158quant_conv2d_4_143160quant_conv2d_4_143162*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *(
_read_only_resource_inputs

*;
config_proto+)

CPU

GPU2*0J
	
   E 8 *S
fNRL
J__inference_quant_conv2d_4_layer_call_and_return_conditional_losses_143151
%quant_max_pooling2d_2/PartitionedCallPartitionedCall/quant_conv2d_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *;
config_proto+)

CPU

GPU2*0J
	
   E 8 *Z
fURS
Q__inference_quant_max_pooling2d_2_layer_call_and_return_conditional_losses_143170ù
quant_flatten/PartitionedCallPartitionedCall.quant_max_pooling2d_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *;
config_proto+)

CPU

GPU2*0J
	
   E 8 *R
fMRK
I__inference_quant_flatten_layer_call_and_return_conditional_losses_143178
#quant_dense/StatefulPartitionedCallStatefulPartitionedCall&quant_flatten/PartitionedCall:output:0quant_dense_143206quant_dense_143208quant_dense_143210quant_dense_143212quant_dense_143214quant_dense_143216*
Tin
	2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*;
config_proto+)

CPU

GPU2*0J
	
   E 8 *P
fKRI
G__inference_quant_dense_layer_call_and_return_conditional_losses_143205÷
quant_dropout/PartitionedCallPartitionedCall,quant_dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *;
config_proto+)

CPU

GPU2*0J
	
   E 8 *R
fMRK
I__inference_quant_dropout_layer_call_and_return_conditional_losses_143224
%quant_dense_1/StatefulPartitionedCallStatefulPartitionedCall&quant_dropout/PartitionedCall:output:0quant_dense_1_143252quant_dense_1_143254quant_dense_1_143256quant_dense_1_143258quant_dense_1_143260quant_dense_1_143262*
Tin
	2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*;
config_proto+)

CPU

GPU2*0J
	
   E 8 *R
fMRK
I__inference_quant_dense_1_layer_call_and_return_conditional_losses_143251ý
quant_dropout_1/PartitionedCallPartitionedCall.quant_dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *;
config_proto+)

CPU

GPU2*0J
	
   E 8 *T
fORM
K__inference_quant_dropout_1_layer_call_and_return_conditional_losses_143270
%quant_dense_2/StatefulPartitionedCallStatefulPartitionedCall(quant_dropout_1/PartitionedCall:output:0quant_dense_2_143294quant_dense_2_143296quant_dense_2_143298quant_dense_2_143300quant_dense_2_143302quant_dense_2_143304*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*;
config_proto+)

CPU

GPU2*0J
	
   E 8 *R
fMRK
I__inference_quant_dense_2_layer_call_and_return_conditional_losses_143293
/conv2d/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpquant_conv2d_142982*&
_output_shapes
: *
dtype0
 conv2d/kernel/Regularizer/L2LossL2Loss7conv2d/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: d
conv2d/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;
conv2d/kernel/Regularizer/mulMul(conv2d/kernel/Regularizer/mul/x:output:0)conv2d/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: 
1conv2d_1/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpquant_conv2d_1_143028*&
_output_shapes
:  *
dtype0
"conv2d_1/kernel/Regularizer/L2LossL2Loss9conv2d_1/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!conv2d_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o; 
conv2d_1/kernel/Regularizer/mulMul*conv2d_1/kernel/Regularizer/mul/x:output:0+conv2d_1/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: 
1conv2d_2/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpquant_conv2d_2_143074*&
_output_shapes
: @*
dtype0
"conv2d_2/kernel/Regularizer/L2LossL2Loss9conv2d_2/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!conv2d_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o; 
conv2d_2/kernel/Regularizer/mulMul*conv2d_2/kernel/Regularizer/mul/x:output:0+conv2d_2/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: 
1conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpquant_conv2d_3_143113*&
_output_shapes
:@@*
dtype0
"conv2d_3/kernel/Regularizer/L2LossL2Loss9conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!conv2d_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o; 
conv2d_3/kernel/Regularizer/mulMul*conv2d_3/kernel/Regularizer/mul/x:output:0+conv2d_3/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: 
1conv2d_4/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpquant_conv2d_4_143152*&
_output_shapes
:@ *
dtype0
"conv2d_4/kernel/Regularizer/L2LossL2Loss9conv2d_4/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!conv2d_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o; 
conv2d_4/kernel/Regularizer/mulMul*conv2d_4/kernel/Regularizer/mul/x:output:0+conv2d_4/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: 
.dense/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpquant_dense_143206* 
_output_shapes
:
*
dtype0
dense/kernel/Regularizer/L2LossL2Loss6dense/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: c
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0(dense/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: 
0dense_1/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpquant_dense_1_143252* 
_output_shapes
:
*
dtype0
!dense_1/kernel/Regularizer/L2LossL2Loss8dense_1/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: e
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0*dense_1/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: }
IdentityIdentity.quant_dense_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp0^conv2d/kernel/Regularizer/L2Loss/ReadVariableOp2^conv2d_1/kernel/Regularizer/L2Loss/ReadVariableOp2^conv2d_2/kernel/Regularizer/L2Loss/ReadVariableOp2^conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOp2^conv2d_4/kernel/Regularizer/L2Loss/ReadVariableOp/^dense/kernel/Regularizer/L2Loss/ReadVariableOp1^dense_1/kernel/Regularizer/L2Loss/ReadVariableOp%^quant_conv2d/StatefulPartitionedCall'^quant_conv2d_1/StatefulPartitionedCall'^quant_conv2d_2/StatefulPartitionedCall'^quant_conv2d_3/StatefulPartitionedCall'^quant_conv2d_4/StatefulPartitionedCall$^quant_dense/StatefulPartitionedCall&^quant_dense_1/StatefulPartitionedCall&^quant_dense_2/StatefulPartitionedCall'^quantize_layer/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:ÿÿÿÿÿÿÿÿÿææ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2b
/conv2d/kernel/Regularizer/L2Loss/ReadVariableOp/conv2d/kernel/Regularizer/L2Loss/ReadVariableOp2f
1conv2d_1/kernel/Regularizer/L2Loss/ReadVariableOp1conv2d_1/kernel/Regularizer/L2Loss/ReadVariableOp2f
1conv2d_2/kernel/Regularizer/L2Loss/ReadVariableOp1conv2d_2/kernel/Regularizer/L2Loss/ReadVariableOp2f
1conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOp1conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOp2f
1conv2d_4/kernel/Regularizer/L2Loss/ReadVariableOp1conv2d_4/kernel/Regularizer/L2Loss/ReadVariableOp2`
.dense/kernel/Regularizer/L2Loss/ReadVariableOp.dense/kernel/Regularizer/L2Loss/ReadVariableOp2d
0dense_1/kernel/Regularizer/L2Loss/ReadVariableOp0dense_1/kernel/Regularizer/L2Loss/ReadVariableOp2L
$quant_conv2d/StatefulPartitionedCall$quant_conv2d/StatefulPartitionedCall2P
&quant_conv2d_1/StatefulPartitionedCall&quant_conv2d_1/StatefulPartitionedCall2P
&quant_conv2d_2/StatefulPartitionedCall&quant_conv2d_2/StatefulPartitionedCall2P
&quant_conv2d_3/StatefulPartitionedCall&quant_conv2d_3/StatefulPartitionedCall2P
&quant_conv2d_4/StatefulPartitionedCall&quant_conv2d_4/StatefulPartitionedCall2J
#quant_dense/StatefulPartitionedCall#quant_dense/StatefulPartitionedCall2N
%quant_dense_1/StatefulPartitionedCall%quant_dense_1/StatefulPartitionedCall2N
%quant_dense_2/StatefulPartitionedCall%quant_dense_2/StatefulPartitionedCall2P
&quantize_layer/StatefulPartitionedCall&quantize_layer/StatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿææ
 
_user_specified_nameinputs
Ù&

I__inference_quant_dense_1_layer_call_and_return_conditional_losses_146947

inputsR
>lastvaluequant_fakequantwithminmaxvars_readvariableop_resource:
J
@lastvaluequant_fakequantwithminmaxvars_readvariableop_1_resource: J
@lastvaluequant_fakequantwithminmaxvars_readvariableop_2_resource: .
biasadd_readvariableop_resource:	K
Amovingavgquantize_fakequantwithminmaxvars_readvariableop_resource: M
Cmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: 
identity¢BiasAdd/ReadVariableOp¢5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp¢7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1¢7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2¢8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp¢:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1¢0dense_1/kernel/Regularizer/L2Loss/ReadVariableOp¶
5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp>lastvaluequant_fakequantwithminmaxvars_readvariableop_resource* 
_output_shapes
:
*
dtype0°
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp@lastvaluequant_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0°
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2ReadVariableOp@lastvaluequant_fakequantwithminmaxvars_readvariableop_2_resource*
_output_shapes
: *
dtype0¸
&LastValueQuant/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars=LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp:value:0?LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0?LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2:value:0* 
_output_shapes
:
*
narrow_range(}
MatMulMatMulinputs0LastValueQuant/FakeQuantWithMinMaxVars:outputs:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ²
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpAmovingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype0¶
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpCmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0
)MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsRelu:activations:0@MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BMovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ±
0dense_1/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp>lastvaluequant_fakequantwithminmaxvars_readvariableop_resource* 
_output_shapes
:
*
dtype0
!dense_1/kernel/Regularizer/L2LossL2Loss8dense_1/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: e
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0*dense_1/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: 
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¶
NoOpNoOp^BiasAdd/ReadVariableOp6^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp8^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_18^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_29^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_11^dense_1/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ: : : : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2n
5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp2r
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_17LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_12r
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_27LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_22t
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12d
0dense_1/kernel/Regularizer/L2Loss/ReadVariableOp0dense_1/kernel/Regularizer/L2Loss/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ÀT
	
I__inference_quant_dense_1_layer_call_and_return_conditional_losses_143619

inputsC
/lastvaluequant_batchmin_readvariableop_resource:
/
%lastvaluequant_assignminlast_resource: /
%lastvaluequant_assignmaxlast_resource: .
biasadd_readvariableop_resource:	@
6movingavgquantize_assignminema_readvariableop_resource: @
6movingavgquantize_assignmaxema_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢LastValueQuant/AssignMaxLast¢LastValueQuant/AssignMinLast¢&LastValueQuant/BatchMax/ReadVariableOp¢&LastValueQuant/BatchMin/ReadVariableOp¢5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp¢7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1¢7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2¢2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp¢-MovingAvgQuantize/AssignMaxEma/ReadVariableOp¢2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp¢-MovingAvgQuantize/AssignMinEma/ReadVariableOp¢8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp¢:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1¢0dense_1/kernel/Regularizer/L2Loss/ReadVariableOpe
LastValueQuant/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
&LastValueQuant/BatchMin/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource* 
_output_shapes
:
*
dtype0
LastValueQuant/BatchMinMin.LastValueQuant/BatchMin/ReadVariableOp:value:0LastValueQuant/Const:output:0*
T0*
_output_shapes
: g
LastValueQuant/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       
&LastValueQuant/BatchMax/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource* 
_output_shapes
:
*
dtype0
LastValueQuant/BatchMaxMax.LastValueQuant/BatchMax/ReadVariableOp:value:0LastValueQuant/Const_1:output:0*
T0*
_output_shapes
: ]
LastValueQuant/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ¿
LastValueQuant/truedivRealDiv LastValueQuant/BatchMax:output:0!LastValueQuant/truediv/y:output:0*
T0*
_output_shapes
: 
LastValueQuant/MinimumMinimum LastValueQuant/BatchMin:output:0LastValueQuant/truediv:z:0*
T0*
_output_shapes
: Y
LastValueQuant/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ¿{
LastValueQuant/mulMul LastValueQuant/BatchMin:output:0LastValueQuant/mul/y:output:0*
T0*
_output_shapes
: |
LastValueQuant/MaximumMaximum LastValueQuant/BatchMax:output:0LastValueQuant/mul:z:0*
T0*
_output_shapes
: ¬
LastValueQuant/AssignMinLastAssignVariableOp%lastvaluequant_assignminlast_resourceLastValueQuant/Minimum:z:0*
_output_shapes
 *
dtype0*
validate_shape(¬
LastValueQuant/AssignMaxLastAssignVariableOp%lastvaluequant_assignmaxlast_resourceLastValueQuant/Maximum:z:0*
_output_shapes
 *
dtype0*
validate_shape(§
5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource* 
_output_shapes
:
*
dtype0´
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp%lastvaluequant_assignminlast_resource^LastValueQuant/AssignMinLast*
_output_shapes
: *
dtype0´
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2ReadVariableOp%lastvaluequant_assignmaxlast_resource^LastValueQuant/AssignMaxLast*
_output_shapes
: *
dtype0¸
&LastValueQuant/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars=LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp:value:0?LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0?LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2:value:0* 
_output_shapes
:
*
narrow_range(}
MatMulMatMulinputs0LastValueQuant/FakeQuantWithMinMaxVars:outputs:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
MovingAvgQuantize/ConstConst*
_output_shapes
:*
dtype0*
valueB"       x
MovingAvgQuantize/BatchMinMinRelu:activations:0 MovingAvgQuantize/Const:output:0*
T0*
_output_shapes
: j
MovingAvgQuantize/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       z
MovingAvgQuantize/BatchMaxMaxRelu:activations:0"MovingAvgQuantize/Const_1:output:0*
T0*
_output_shapes
: `
MovingAvgQuantize/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
MovingAvgQuantize/MinimumMinimum#MovingAvgQuantize/BatchMin:output:0$MovingAvgQuantize/Minimum/y:output:0*
T0*
_output_shapes
: `
MovingAvgQuantize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
MovingAvgQuantize/MaximumMaximum#MovingAvgQuantize/BatchMax:output:0$MovingAvgQuantize/Maximum/y:output:0*
T0*
_output_shapes
: i
$MovingAvgQuantize/AssignMinEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o:
-MovingAvgQuantize/AssignMinEma/ReadVariableOpReadVariableOp6movingavgquantize_assignminema_readvariableop_resource*
_output_shapes
: *
dtype0 
"MovingAvgQuantize/AssignMinEma/subSub5MovingAvgQuantize/AssignMinEma/ReadVariableOp:value:0MovingAvgQuantize/Minimum:z:0*
T0*
_output_shapes
: ¡
"MovingAvgQuantize/AssignMinEma/mulMul&MovingAvgQuantize/AssignMinEma/sub:z:0-MovingAvgQuantize/AssignMinEma/decay:output:0*
T0*
_output_shapes
: ü
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOpAssignSubVariableOp6movingavgquantize_assignminema_readvariableop_resource&MovingAvgQuantize/AssignMinEma/mul:z:0.^MovingAvgQuantize/AssignMinEma/ReadVariableOp*
_output_shapes
 *
dtype0i
$MovingAvgQuantize/AssignMaxEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o:
-MovingAvgQuantize/AssignMaxEma/ReadVariableOpReadVariableOp6movingavgquantize_assignmaxema_readvariableop_resource*
_output_shapes
: *
dtype0 
"MovingAvgQuantize/AssignMaxEma/subSub5MovingAvgQuantize/AssignMaxEma/ReadVariableOp:value:0MovingAvgQuantize/Maximum:z:0*
T0*
_output_shapes
: ¡
"MovingAvgQuantize/AssignMaxEma/mulMul&MovingAvgQuantize/AssignMaxEma/sub:z:0-MovingAvgQuantize/AssignMaxEma/decay:output:0*
T0*
_output_shapes
: ü
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpAssignSubVariableOp6movingavgquantize_assignmaxema_readvariableop_resource&MovingAvgQuantize/AssignMaxEma/mul:z:0.^MovingAvgQuantize/AssignMaxEma/ReadVariableOp*
_output_shapes
 *
dtype0Ü
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp6movingavgquantize_assignminema_readvariableop_resource3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp*
_output_shapes
: *
dtype0Þ
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp6movingavgquantize_assignmaxema_readvariableop_resource3^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp*
_output_shapes
: *
dtype0
)MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsRelu:activations:0@MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BMovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢
0dense_1/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource* 
_output_shapes
:
*
dtype0
!dense_1/kernel/Regularizer/L2LossL2Loss8dense_1/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: e
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0*dense_1/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: 
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp^BiasAdd/ReadVariableOp^LastValueQuant/AssignMaxLast^LastValueQuant/AssignMinLast'^LastValueQuant/BatchMax/ReadVariableOp'^LastValueQuant/BatchMin/ReadVariableOp6^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp8^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_18^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_23^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp.^MovingAvgQuantize/AssignMaxEma/ReadVariableOp3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp.^MovingAvgQuantize/AssignMinEma/ReadVariableOp9^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_11^dense_1/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ: : : : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2<
LastValueQuant/AssignMaxLastLastValueQuant/AssignMaxLast2<
LastValueQuant/AssignMinLastLastValueQuant/AssignMinLast2P
&LastValueQuant/BatchMax/ReadVariableOp&LastValueQuant/BatchMax/ReadVariableOp2P
&LastValueQuant/BatchMin/ReadVariableOp&LastValueQuant/BatchMin/ReadVariableOp2n
5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp2r
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_17LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_12r
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_27LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_22h
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2^
-MovingAvgQuantize/AssignMaxEma/ReadVariableOp-MovingAvgQuantize/AssignMaxEma/ReadVariableOp2h
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2^
-MovingAvgQuantize/AssignMinEma/ReadVariableOp-MovingAvgQuantize/AssignMinEma/ReadVariableOp2t
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12d
0dense_1/kernel/Regularizer/L2Loss/ReadVariableOp0dense_1/kernel/Regularizer/L2Loss/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
#
ý
J__inference_quantize_layer_layer_call_and_return_conditional_losses_146083

inputs;
1allvaluesquantize_minimum_readvariableop_resource: ;
1allvaluesquantize_maximum_readvariableop_resource: 
identity¢#AllValuesQuantize/AssignMaxAllValue¢#AllValuesQuantize/AssignMinAllValue¢8AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp¢:AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1¢(AllValuesQuantize/Maximum/ReadVariableOp¢(AllValuesQuantize/Minimum/ReadVariableOpp
AllValuesQuantize/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             l
AllValuesQuantize/BatchMinMininputs AllValuesQuantize/Const:output:0*
T0*
_output_shapes
: r
AllValuesQuantize/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             n
AllValuesQuantize/BatchMaxMaxinputs"AllValuesQuantize/Const_1:output:0*
T0*
_output_shapes
: 
(AllValuesQuantize/Minimum/ReadVariableOpReadVariableOp1allvaluesquantize_minimum_readvariableop_resource*
_output_shapes
: *
dtype0
AllValuesQuantize/MinimumMinimum0AllValuesQuantize/Minimum/ReadVariableOp:value:0#AllValuesQuantize/BatchMin:output:0*
T0*
_output_shapes
: b
AllValuesQuantize/Minimum_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
AllValuesQuantize/Minimum_1MinimumAllValuesQuantize/Minimum:z:0&AllValuesQuantize/Minimum_1/y:output:0*
T0*
_output_shapes
: 
(AllValuesQuantize/Maximum/ReadVariableOpReadVariableOp1allvaluesquantize_maximum_readvariableop_resource*
_output_shapes
: *
dtype0
AllValuesQuantize/MaximumMaximum0AllValuesQuantize/Maximum/ReadVariableOp:value:0#AllValuesQuantize/BatchMax:output:0*
T0*
_output_shapes
: b
AllValuesQuantize/Maximum_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
AllValuesQuantize/Maximum_1MaximumAllValuesQuantize/Maximum:z:0&AllValuesQuantize/Maximum_1/y:output:0*
T0*
_output_shapes
: ï
#AllValuesQuantize/AssignMinAllValueAssignVariableOp1allvaluesquantize_minimum_readvariableop_resourceAllValuesQuantize/Minimum_1:z:0)^AllValuesQuantize/Minimum/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(ï
#AllValuesQuantize/AssignMaxAllValueAssignVariableOp1allvaluesquantize_maximum_readvariableop_resourceAllValuesQuantize/Maximum_1:z:0)^AllValuesQuantize/Maximum/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(È
8AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp1allvaluesquantize_minimum_readvariableop_resource$^AllValuesQuantize/AssignMinAllValue*
_output_shapes
: *
dtype0Ê
:AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp1allvaluesquantize_maximum_readvariableop_resource$^AllValuesQuantize/AssignMaxAllValue*
_output_shapes
: *
dtype0
)AllValuesQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsinputs@AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BAllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿææ
IdentityIdentity3AllValuesQuantize/FakeQuantWithMinMaxVars:outputs:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿææà
NoOpNoOp$^AllValuesQuantize/AssignMaxAllValue$^AllValuesQuantize/AssignMinAllValue9^AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1)^AllValuesQuantize/Maximum/ReadVariableOp)^AllValuesQuantize/Minimum/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿææ: : 2J
#AllValuesQuantize/AssignMaxAllValue#AllValuesQuantize/AssignMaxAllValue2J
#AllValuesQuantize/AssignMinAllValue#AllValuesQuantize/AssignMinAllValue2t
8AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12T
(AllValuesQuantize/Maximum/ReadVariableOp(AllValuesQuantize/Maximum/ReadVariableOp2T
(AllValuesQuantize/Minimum/ReadVariableOp(AllValuesQuantize/Minimum/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿææ
 
_user_specified_nameinputs
â
i
K__inference_quant_dropout_1_layer_call_and_return_conditional_losses_147019

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
­
m
Q__inference_quant_max_pooling2d_2_layer_call_and_return_conditional_losses_143775

inputs
identity
MaxPoolMaxPoolinputs*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
ksize
*
paddingVALID*
strides
`
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ :W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
ö)
×
H__inference_quant_conv2d_layer_call_and_return_conditional_losses_142981

inputsb
Hlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource: X
Jlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource: X
Jlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource: -
biasadd_readvariableop_resource: K
Amovingavgquantize_fakequantwithminmaxvars_readvariableop_resource: M
Cmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: 
identity¢BiasAdd/ReadVariableOp¢?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp¢ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1¢ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2¢8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp¢:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1¢/conv2d/kernel/Regularizer/L2Loss/ReadVariableOpÐ
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpReadVariableOpHlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource*&
_output_shapes
: *
dtype0È
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ReadVariableOpJlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource*
_output_shapes
: *
dtype0È
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ReadVariableOpJlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource*
_output_shapes
: *
dtype0ð
0LastValueQuant/FakeQuantWithMinMaxVarsPerChannel!FakeQuantWithMinMaxVarsPerChannelGLastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2:value:0*&
_output_shapes
: *
narrow_range(¹
Conv2DConv2Dinputs:LastValueQuant/FakeQuantWithMinMaxVarsPerChannel:outputs:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿââ *
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿââ Z
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿââ ²
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpAmovingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype0¶
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpCmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0
)MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsRelu:activations:0@MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BMovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿââ À
/conv2d/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpHlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource*&
_output_shapes
: *
dtype0
 conv2d/kernel/Regularizer/L2LossL2Loss7conv2d/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: d
conv2d/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;
conv2d/kernel/Regularizer/mulMul(conv2d/kernel/Regularizer/mul/x:output:0)conv2d/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: 
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿââ Ó
NoOpNoOp^BiasAdd/ReadVariableOp@^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpB^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1B^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_29^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_10^conv2d/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):ÿÿÿÿÿÿÿÿÿææ: : : : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp2
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_12
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_22t
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12b
/conv2d/kernel/Regularizer/L2Loss/ReadVariableOp/conv2d/kernel/Regularizer/L2Loss/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿææ
 
_user_specified_nameinputs

g
.__inference_quant_dropout_layer_call_fn_146871

inputs
identity¢StatefulPartitionedCallÓ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *;
config_proto+)

CPU

GPU2*0J
	
   E 8 *R
fMRK
I__inference_quant_dropout_layer_call_and_return_conditional_losses_143654p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
é
ð
,__inference_quant_dense_layer_call_fn_146779

inputs
unknown:

	unknown_0: 
	unknown_1: 
	unknown_2:	
	unknown_3: 
	unknown_4: 
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*;
config_proto+)

CPU

GPU2*0J
	
   E 8 *P
fKRI
G__inference_quant_dense_layer_call_and_return_conditional_losses_143730p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
­X
º	
J__inference_quant_conv2d_1_layer_call_and_return_conditional_losses_146331

inputsI
/lastvaluequant_batchmin_readvariableop_resource:  3
%lastvaluequant_assignminlast_resource: 3
%lastvaluequant_assignmaxlast_resource: -
biasadd_readvariableop_resource: @
6movingavgquantize_assignminema_readvariableop_resource: @
6movingavgquantize_assignmaxema_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢LastValueQuant/AssignMaxLast¢LastValueQuant/AssignMinLast¢&LastValueQuant/BatchMax/ReadVariableOp¢&LastValueQuant/BatchMin/ReadVariableOp¢?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp¢ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1¢ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2¢2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp¢-MovingAvgQuantize/AssignMaxEma/ReadVariableOp¢2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp¢-MovingAvgQuantize/AssignMinEma/ReadVariableOp¢8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp¢:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1¢1conv2d_1/kernel/Regularizer/L2Loss/ReadVariableOp
&LastValueQuant/BatchMin/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:  *
dtype0~
)LastValueQuant/BatchMin/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          §
LastValueQuant/BatchMinMin.LastValueQuant/BatchMin/ReadVariableOp:value:02LastValueQuant/BatchMin/reduction_indices:output:0*
T0*
_output_shapes
: 
&LastValueQuant/BatchMax/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:  *
dtype0~
)LastValueQuant/BatchMax/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          §
LastValueQuant/BatchMaxMax.LastValueQuant/BatchMax/ReadVariableOp:value:02LastValueQuant/BatchMax/reduction_indices:output:0*
T0*
_output_shapes
: ]
LastValueQuant/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ¿
LastValueQuant/truedivRealDiv LastValueQuant/BatchMax:output:0!LastValueQuant/truediv/y:output:0*
T0*
_output_shapes
: 
LastValueQuant/MinimumMinimum LastValueQuant/BatchMin:output:0LastValueQuant/truediv:z:0*
T0*
_output_shapes
: Y
LastValueQuant/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ¿
LastValueQuant/mulMul LastValueQuant/BatchMin:output:0LastValueQuant/mul/y:output:0*
T0*
_output_shapes
: 
LastValueQuant/MaximumMaximum LastValueQuant/BatchMax:output:0LastValueQuant/mul:z:0*
T0*
_output_shapes
: ¬
LastValueQuant/AssignMinLastAssignVariableOp%lastvaluequant_assignminlast_resourceLastValueQuant/Minimum:z:0*
_output_shapes
 *
dtype0*
validate_shape(¬
LastValueQuant/AssignMaxLastAssignVariableOp%lastvaluequant_assignmaxlast_resourceLastValueQuant/Maximum:z:0*
_output_shapes
 *
dtype0*
validate_shape(·
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:  *
dtype0Â
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ReadVariableOp%lastvaluequant_assignminlast_resource^LastValueQuant/AssignMinLast*
_output_shapes
: *
dtype0Â
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ReadVariableOp%lastvaluequant_assignmaxlast_resource^LastValueQuant/AssignMaxLast*
_output_shapes
: *
dtype0ð
0LastValueQuant/FakeQuantWithMinMaxVarsPerChannel!FakeQuantWithMinMaxVarsPerChannelGLastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2:value:0*&
_output_shapes
:  *
narrow_range(¶
Conv2DConv2Dinputs:LastValueQuant/FakeQuantWithMinMaxVarsPerChannel:outputs:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿKK *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿKK X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿKK p
MovingAvgQuantize/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             x
MovingAvgQuantize/BatchMinMinRelu:activations:0 MovingAvgQuantize/Const:output:0*
T0*
_output_shapes
: r
MovingAvgQuantize/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             z
MovingAvgQuantize/BatchMaxMaxRelu:activations:0"MovingAvgQuantize/Const_1:output:0*
T0*
_output_shapes
: `
MovingAvgQuantize/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
MovingAvgQuantize/MinimumMinimum#MovingAvgQuantize/BatchMin:output:0$MovingAvgQuantize/Minimum/y:output:0*
T0*
_output_shapes
: `
MovingAvgQuantize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
MovingAvgQuantize/MaximumMaximum#MovingAvgQuantize/BatchMax:output:0$MovingAvgQuantize/Maximum/y:output:0*
T0*
_output_shapes
: i
$MovingAvgQuantize/AssignMinEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o:
-MovingAvgQuantize/AssignMinEma/ReadVariableOpReadVariableOp6movingavgquantize_assignminema_readvariableop_resource*
_output_shapes
: *
dtype0 
"MovingAvgQuantize/AssignMinEma/subSub5MovingAvgQuantize/AssignMinEma/ReadVariableOp:value:0MovingAvgQuantize/Minimum:z:0*
T0*
_output_shapes
: ¡
"MovingAvgQuantize/AssignMinEma/mulMul&MovingAvgQuantize/AssignMinEma/sub:z:0-MovingAvgQuantize/AssignMinEma/decay:output:0*
T0*
_output_shapes
: ü
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOpAssignSubVariableOp6movingavgquantize_assignminema_readvariableop_resource&MovingAvgQuantize/AssignMinEma/mul:z:0.^MovingAvgQuantize/AssignMinEma/ReadVariableOp*
_output_shapes
 *
dtype0i
$MovingAvgQuantize/AssignMaxEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o:
-MovingAvgQuantize/AssignMaxEma/ReadVariableOpReadVariableOp6movingavgquantize_assignmaxema_readvariableop_resource*
_output_shapes
: *
dtype0 
"MovingAvgQuantize/AssignMaxEma/subSub5MovingAvgQuantize/AssignMaxEma/ReadVariableOp:value:0MovingAvgQuantize/Maximum:z:0*
T0*
_output_shapes
: ¡
"MovingAvgQuantize/AssignMaxEma/mulMul&MovingAvgQuantize/AssignMaxEma/sub:z:0-MovingAvgQuantize/AssignMaxEma/decay:output:0*
T0*
_output_shapes
: ü
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpAssignSubVariableOp6movingavgquantize_assignmaxema_readvariableop_resource&MovingAvgQuantize/AssignMaxEma/mul:z:0.^MovingAvgQuantize/AssignMaxEma/ReadVariableOp*
_output_shapes
 *
dtype0Ü
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp6movingavgquantize_assignminema_readvariableop_resource3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp*
_output_shapes
: *
dtype0Þ
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp6movingavgquantize_assignmaxema_readvariableop_resource3^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp*
_output_shapes
: *
dtype0
)MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsRelu:activations:0@MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BMovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿKK ©
1conv2d_1/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:  *
dtype0
"conv2d_1/kernel/Regularizer/L2LossL2Loss9conv2d_1/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!conv2d_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o; 
conv2d_1/kernel/Regularizer/mulMul*conv2d_1/kernel/Regularizer/mul/x:output:0+conv2d_1/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: 
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿKK ¯
NoOpNoOp^BiasAdd/ReadVariableOp^LastValueQuant/AssignMaxLast^LastValueQuant/AssignMinLast'^LastValueQuant/BatchMax/ReadVariableOp'^LastValueQuant/BatchMin/ReadVariableOp@^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpB^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1B^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_23^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp.^MovingAvgQuantize/AssignMaxEma/ReadVariableOp3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp.^MovingAvgQuantize/AssignMinEma/ReadVariableOp9^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12^conv2d_1/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿKK : : : : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2<
LastValueQuant/AssignMaxLastLastValueQuant/AssignMaxLast2<
LastValueQuant/AssignMinLastLastValueQuant/AssignMinLast2P
&LastValueQuant/BatchMax/ReadVariableOp&LastValueQuant/BatchMax/ReadVariableOp2P
&LastValueQuant/BatchMin/ReadVariableOp&LastValueQuant/BatchMin/ReadVariableOp2
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp2
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_12
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_22h
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2^
-MovingAvgQuantize/AssignMaxEma/ReadVariableOp-MovingAvgQuantize/AssignMaxEma/ReadVariableOp2h
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2^
-MovingAvgQuantize/AssignMinEma/ReadVariableOp-MovingAvgQuantize/AssignMinEma/ReadVariableOp2t
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12f
1conv2d_1/kernel/Regularizer/L2Loss/ReadVariableOp1conv2d_1/kernel/Regularizer/L2Loss/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿKK 
 
_user_specified_nameinputs
°	
¹
__inference_loss_fn_3_147191T
:conv2d_3_kernel_regularizer_l2loss_readvariableop_resource:@@
identity¢1conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOp´
1conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp:conv2d_3_kernel_regularizer_l2loss_readvariableop_resource*&
_output_shapes
:@@*
dtype0
"conv2d_3/kernel/Regularizer/L2LossL2Loss9conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!conv2d_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o; 
conv2d_3/kernel/Regularizer/mulMul*conv2d_3/kernel/Regularizer/mul/x:output:0+conv2d_3/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: a
IdentityIdentity#conv2d_3/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: z
NoOpNoOp2^conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2f
1conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOp1conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOp
ð~
º
F__inference_sequential_layer_call_and_return_conditional_losses_144899
conv2d_input
quantize_layer_144755: 
quantize_layer_144757: -
quant_conv2d_144760: !
quant_conv2d_144762: !
quant_conv2d_144764: !
quant_conv2d_144766: 
quant_conv2d_144768: 
quant_conv2d_144770: /
quant_conv2d_1_144774:  #
quant_conv2d_1_144776: #
quant_conv2d_1_144778: #
quant_conv2d_1_144780: 
quant_conv2d_1_144782: 
quant_conv2d_1_144784: /
quant_conv2d_2_144788: @#
quant_conv2d_2_144790:@#
quant_conv2d_2_144792:@#
quant_conv2d_2_144794:@
quant_conv2d_2_144796: 
quant_conv2d_2_144798: /
quant_conv2d_3_144801:@@#
quant_conv2d_3_144803:@#
quant_conv2d_3_144805:@#
quant_conv2d_3_144807:@
quant_conv2d_3_144809: 
quant_conv2d_3_144811: /
quant_conv2d_4_144814:@ #
quant_conv2d_4_144816: #
quant_conv2d_4_144818: #
quant_conv2d_4_144820: 
quant_conv2d_4_144822: 
quant_conv2d_4_144824: &
quant_dense_144829:

quant_dense_144831: 
quant_dense_144833: !
quant_dense_144835:	
quant_dense_144837: 
quant_dense_144839: (
quant_dense_1_144843:

quant_dense_1_144845: 
quant_dense_1_144847: #
quant_dense_1_144849:	
quant_dense_1_144851: 
quant_dense_1_144853: '
quant_dense_2_144857:	
quant_dense_2_144859: 
quant_dense_2_144861: "
quant_dense_2_144863:
quant_dense_2_144865: 
quant_dense_2_144867: 
identity¢/conv2d/kernel/Regularizer/L2Loss/ReadVariableOp¢1conv2d_1/kernel/Regularizer/L2Loss/ReadVariableOp¢1conv2d_2/kernel/Regularizer/L2Loss/ReadVariableOp¢1conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOp¢1conv2d_4/kernel/Regularizer/L2Loss/ReadVariableOp¢.dense/kernel/Regularizer/L2Loss/ReadVariableOp¢0dense_1/kernel/Regularizer/L2Loss/ReadVariableOp¢$quant_conv2d/StatefulPartitionedCall¢&quant_conv2d_1/StatefulPartitionedCall¢&quant_conv2d_2/StatefulPartitionedCall¢&quant_conv2d_3/StatefulPartitionedCall¢&quant_conv2d_4/StatefulPartitionedCall¢#quant_dense/StatefulPartitionedCall¢%quant_dense_1/StatefulPartitionedCall¢%quant_dense_2/StatefulPartitionedCall¢&quantize_layer/StatefulPartitionedCall¦
&quantize_layer/StatefulPartitionedCallStatefulPartitionedCallconv2d_inputquantize_layer_144755quantize_layer_144757*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿææ*$
_read_only_resource_inputs
*;
config_proto+)

CPU

GPU2*0J
	
   E 8 *S
fNRL
J__inference_quantize_layer_layer_call_and_return_conditional_losses_142950
$quant_conv2d/StatefulPartitionedCallStatefulPartitionedCall/quantize_layer/StatefulPartitionedCall:output:0quant_conv2d_144760quant_conv2d_144762quant_conv2d_144764quant_conv2d_144766quant_conv2d_144768quant_conv2d_144770*
Tin
	2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿââ *(
_read_only_resource_inputs

*;
config_proto+)

CPU

GPU2*0J
	
   E 8 *Q
fLRJ
H__inference_quant_conv2d_layer_call_and_return_conditional_losses_142981
#quant_max_pooling2d/PartitionedCallPartitionedCall-quant_conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿKK * 
_read_only_resource_inputs
 *;
config_proto+)

CPU

GPU2*0J
	
   E 8 *X
fSRQ
O__inference_quant_max_pooling2d_layer_call_and_return_conditional_losses_143000¨
&quant_conv2d_1/StatefulPartitionedCallStatefulPartitionedCall,quant_max_pooling2d/PartitionedCall:output:0quant_conv2d_1_144774quant_conv2d_1_144776quant_conv2d_1_144778quant_conv2d_1_144780quant_conv2d_1_144782quant_conv2d_1_144784*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿKK *(
_read_only_resource_inputs

*;
config_proto+)

CPU

GPU2*0J
	
   E 8 *S
fNRL
J__inference_quant_conv2d_1_layer_call_and_return_conditional_losses_143027
%quant_max_pooling2d_1/PartitionedCallPartitionedCall/quant_conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *;
config_proto+)

CPU

GPU2*0J
	
   E 8 *Z
fURS
Q__inference_quant_max_pooling2d_1_layer_call_and_return_conditional_losses_143046ª
&quant_conv2d_2/StatefulPartitionedCallStatefulPartitionedCall.quant_max_pooling2d_1/PartitionedCall:output:0quant_conv2d_2_144788quant_conv2d_2_144790quant_conv2d_2_144792quant_conv2d_2_144794quant_conv2d_2_144796quant_conv2d_2_144798*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*(
_read_only_resource_inputs

*;
config_proto+)

CPU

GPU2*0J
	
   E 8 *S
fNRL
J__inference_quant_conv2d_2_layer_call_and_return_conditional_losses_143073«
&quant_conv2d_3/StatefulPartitionedCallStatefulPartitionedCall/quant_conv2d_2/StatefulPartitionedCall:output:0quant_conv2d_3_144801quant_conv2d_3_144803quant_conv2d_3_144805quant_conv2d_3_144807quant_conv2d_3_144809quant_conv2d_3_144811*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*(
_read_only_resource_inputs

*;
config_proto+)

CPU

GPU2*0J
	
   E 8 *S
fNRL
J__inference_quant_conv2d_3_layer_call_and_return_conditional_losses_143112«
&quant_conv2d_4/StatefulPartitionedCallStatefulPartitionedCall/quant_conv2d_3/StatefulPartitionedCall:output:0quant_conv2d_4_144814quant_conv2d_4_144816quant_conv2d_4_144818quant_conv2d_4_144820quant_conv2d_4_144822quant_conv2d_4_144824*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *(
_read_only_resource_inputs

*;
config_proto+)

CPU

GPU2*0J
	
   E 8 *S
fNRL
J__inference_quant_conv2d_4_layer_call_and_return_conditional_losses_143151
%quant_max_pooling2d_2/PartitionedCallPartitionedCall/quant_conv2d_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *;
config_proto+)

CPU

GPU2*0J
	
   E 8 *Z
fURS
Q__inference_quant_max_pooling2d_2_layer_call_and_return_conditional_losses_143170ù
quant_flatten/PartitionedCallPartitionedCall.quant_max_pooling2d_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *;
config_proto+)

CPU

GPU2*0J
	
   E 8 *R
fMRK
I__inference_quant_flatten_layer_call_and_return_conditional_losses_143178
#quant_dense/StatefulPartitionedCallStatefulPartitionedCall&quant_flatten/PartitionedCall:output:0quant_dense_144829quant_dense_144831quant_dense_144833quant_dense_144835quant_dense_144837quant_dense_144839*
Tin
	2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*;
config_proto+)

CPU

GPU2*0J
	
   E 8 *P
fKRI
G__inference_quant_dense_layer_call_and_return_conditional_losses_143205÷
quant_dropout/PartitionedCallPartitionedCall,quant_dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *;
config_proto+)

CPU

GPU2*0J
	
   E 8 *R
fMRK
I__inference_quant_dropout_layer_call_and_return_conditional_losses_143224
%quant_dense_1/StatefulPartitionedCallStatefulPartitionedCall&quant_dropout/PartitionedCall:output:0quant_dense_1_144843quant_dense_1_144845quant_dense_1_144847quant_dense_1_144849quant_dense_1_144851quant_dense_1_144853*
Tin
	2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*;
config_proto+)

CPU

GPU2*0J
	
   E 8 *R
fMRK
I__inference_quant_dense_1_layer_call_and_return_conditional_losses_143251ý
quant_dropout_1/PartitionedCallPartitionedCall.quant_dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *;
config_proto+)

CPU

GPU2*0J
	
   E 8 *T
fORM
K__inference_quant_dropout_1_layer_call_and_return_conditional_losses_143270
%quant_dense_2/StatefulPartitionedCallStatefulPartitionedCall(quant_dropout_1/PartitionedCall:output:0quant_dense_2_144857quant_dense_2_144859quant_dense_2_144861quant_dense_2_144863quant_dense_2_144865quant_dense_2_144867*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*;
config_proto+)

CPU

GPU2*0J
	
   E 8 *R
fMRK
I__inference_quant_dense_2_layer_call_and_return_conditional_losses_143293
/conv2d/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpquant_conv2d_144760*&
_output_shapes
: *
dtype0
 conv2d/kernel/Regularizer/L2LossL2Loss7conv2d/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: d
conv2d/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;
conv2d/kernel/Regularizer/mulMul(conv2d/kernel/Regularizer/mul/x:output:0)conv2d/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: 
1conv2d_1/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpquant_conv2d_1_144774*&
_output_shapes
:  *
dtype0
"conv2d_1/kernel/Regularizer/L2LossL2Loss9conv2d_1/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!conv2d_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o; 
conv2d_1/kernel/Regularizer/mulMul*conv2d_1/kernel/Regularizer/mul/x:output:0+conv2d_1/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: 
1conv2d_2/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpquant_conv2d_2_144788*&
_output_shapes
: @*
dtype0
"conv2d_2/kernel/Regularizer/L2LossL2Loss9conv2d_2/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!conv2d_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o; 
conv2d_2/kernel/Regularizer/mulMul*conv2d_2/kernel/Regularizer/mul/x:output:0+conv2d_2/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: 
1conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpquant_conv2d_3_144801*&
_output_shapes
:@@*
dtype0
"conv2d_3/kernel/Regularizer/L2LossL2Loss9conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!conv2d_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o; 
conv2d_3/kernel/Regularizer/mulMul*conv2d_3/kernel/Regularizer/mul/x:output:0+conv2d_3/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: 
1conv2d_4/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpquant_conv2d_4_144814*&
_output_shapes
:@ *
dtype0
"conv2d_4/kernel/Regularizer/L2LossL2Loss9conv2d_4/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!conv2d_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o; 
conv2d_4/kernel/Regularizer/mulMul*conv2d_4/kernel/Regularizer/mul/x:output:0+conv2d_4/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: 
.dense/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpquant_dense_144829* 
_output_shapes
:
*
dtype0
dense/kernel/Regularizer/L2LossL2Loss6dense/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: c
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0(dense/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: 
0dense_1/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpquant_dense_1_144843* 
_output_shapes
:
*
dtype0
!dense_1/kernel/Regularizer/L2LossL2Loss8dense_1/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: e
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0*dense_1/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: }
IdentityIdentity.quant_dense_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp0^conv2d/kernel/Regularizer/L2Loss/ReadVariableOp2^conv2d_1/kernel/Regularizer/L2Loss/ReadVariableOp2^conv2d_2/kernel/Regularizer/L2Loss/ReadVariableOp2^conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOp2^conv2d_4/kernel/Regularizer/L2Loss/ReadVariableOp/^dense/kernel/Regularizer/L2Loss/ReadVariableOp1^dense_1/kernel/Regularizer/L2Loss/ReadVariableOp%^quant_conv2d/StatefulPartitionedCall'^quant_conv2d_1/StatefulPartitionedCall'^quant_conv2d_2/StatefulPartitionedCall'^quant_conv2d_3/StatefulPartitionedCall'^quant_conv2d_4/StatefulPartitionedCall$^quant_dense/StatefulPartitionedCall&^quant_dense_1/StatefulPartitionedCall&^quant_dense_2/StatefulPartitionedCall'^quantize_layer/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:ÿÿÿÿÿÿÿÿÿææ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2b
/conv2d/kernel/Regularizer/L2Loss/ReadVariableOp/conv2d/kernel/Regularizer/L2Loss/ReadVariableOp2f
1conv2d_1/kernel/Regularizer/L2Loss/ReadVariableOp1conv2d_1/kernel/Regularizer/L2Loss/ReadVariableOp2f
1conv2d_2/kernel/Regularizer/L2Loss/ReadVariableOp1conv2d_2/kernel/Regularizer/L2Loss/ReadVariableOp2f
1conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOp1conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOp2f
1conv2d_4/kernel/Regularizer/L2Loss/ReadVariableOp1conv2d_4/kernel/Regularizer/L2Loss/ReadVariableOp2`
.dense/kernel/Regularizer/L2Loss/ReadVariableOp.dense/kernel/Regularizer/L2Loss/ReadVariableOp2d
0dense_1/kernel/Regularizer/L2Loss/ReadVariableOp0dense_1/kernel/Regularizer/L2Loss/ReadVariableOp2L
$quant_conv2d/StatefulPartitionedCall$quant_conv2d/StatefulPartitionedCall2P
&quant_conv2d_1/StatefulPartitionedCall&quant_conv2d_1/StatefulPartitionedCall2P
&quant_conv2d_2/StatefulPartitionedCall&quant_conv2d_2/StatefulPartitionedCall2P
&quant_conv2d_3/StatefulPartitionedCall&quant_conv2d_3/StatefulPartitionedCall2P
&quant_conv2d_4/StatefulPartitionedCall&quant_conv2d_4/StatefulPartitionedCall2J
#quant_dense/StatefulPartitionedCall#quant_dense/StatefulPartitionedCall2N
%quant_dense_1/StatefulPartitionedCall%quant_dense_1/StatefulPartitionedCall2N
%quant_dense_2/StatefulPartitionedCall%quant_dense_2/StatefulPartitionedCall2P
&quantize_layer/StatefulPartitionedCall&quantize_layer/StatefulPartitionedCall:_ [
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿææ
&
_user_specified_nameconv2d_input
	
±
__inference_loss_fn_6_147228M
9dense_1_kernel_regularizer_l2loss_readvariableop_resource:

identity¢0dense_1/kernel/Regularizer/L2Loss/ReadVariableOp¬
0dense_1/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp9dense_1_kernel_regularizer_l2loss_readvariableop_resource* 
_output_shapes
:
*
dtype0
!dense_1/kernel/Regularizer/L2LossL2Loss8dense_1/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: e
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0*dense_1/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: `
IdentityIdentity"dense_1/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: y
NoOpNoOp1^dense_1/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2d
0dense_1/kernel/Regularizer/L2Loss/ReadVariableOp0dense_1/kernel/Regularizer/L2Loss/ReadVariableOp
Ë
e
I__inference_quant_flatten_layer_call_and_return_conditional_losses_146739

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ :W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
å
R
6__inference_quant_max_pooling2d_2_layer_call_fn_146708

inputs
identityÒ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *;
config_proto+)

CPU

GPU2*0J
	
   E 8 *Z
fURS
Q__inference_quant_max_pooling2d_2_layer_call_and_return_conditional_losses_143170h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ :W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
°	
¹
__inference_loss_fn_1_147163T
:conv2d_1_kernel_regularizer_l2loss_readvariableop_resource:  
identity¢1conv2d_1/kernel/Regularizer/L2Loss/ReadVariableOp´
1conv2d_1/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp:conv2d_1_kernel_regularizer_l2loss_readvariableop_resource*&
_output_shapes
:  *
dtype0
"conv2d_1/kernel/Regularizer/L2LossL2Loss9conv2d_1/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!conv2d_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o; 
conv2d_1/kernel/Regularizer/mulMul*conv2d_1/kernel/Regularizer/mul/x:output:0+conv2d_1/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: a
IdentityIdentity#conv2d_1/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: z
NoOpNoOp2^conv2d_1/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2f
1conv2d_1/kernel/Regularizer/L2Loss/ReadVariableOp1conv2d_1/kernel/Regularizer/L2Loss/ReadVariableOp
	

/__inference_quant_conv2d_4_layer_call_fn_146621

inputs!
unknown:@ 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: 
identity¢StatefulPartitionedCall¥
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*;
config_proto+)

CPU

GPU2*0J
	
   E 8 *S
fNRL
J__inference_quant_conv2d_4_layer_call_and_return_conditional_losses_143851w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ@: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
ÿ)
Û
J__inference_quant_conv2d_3_layer_call_and_return_conditional_losses_143112

inputsb
Hlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource:@@X
Jlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource:@X
Jlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource:@-
biasadd_readvariableop_resource:@K
Amovingavgquantize_fakequantwithminmaxvars_readvariableop_resource: M
Cmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: 
identity¢BiasAdd/ReadVariableOp¢?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp¢ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1¢ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2¢8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp¢:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1¢1conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOpÐ
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpReadVariableOpHlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource*&
_output_shapes
:@@*
dtype0È
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ReadVariableOpJlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource*
_output_shapes
:@*
dtype0È
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ReadVariableOpJlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource*
_output_shapes
:@*
dtype0ð
0LastValueQuant/FakeQuantWithMinMaxVarsPerChannel!FakeQuantWithMinMaxVarsPerChannelGLastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2:value:0*&
_output_shapes
:@@*
narrow_range(¶
Conv2DConv2Dinputs:LastValueQuant/FakeQuantWithMinMaxVarsPerChannel:outputs:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@²
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpAmovingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype0¶
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpCmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0
)MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsRelu:activations:0@MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BMovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Â
1conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpHlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource*&
_output_shapes
:@@*
dtype0
"conv2d_3/kernel/Regularizer/L2LossL2Loss9conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!conv2d_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o; 
conv2d_3/kernel/Regularizer/mulMul*conv2d_3/kernel/Regularizer/mul/x:output:0+conv2d_3/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: 
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Õ
NoOpNoOp^BiasAdd/ReadVariableOp@^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpB^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1B^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_29^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12^conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ@: : : : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp2
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_12
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_22t
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12f
1conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOp1conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
 
ªC
"__inference__traced_restore_147897
file_prefix<
2assignvariableop_quantize_layer_quantize_layer_min: >
4assignvariableop_1_quantize_layer_quantize_layer_max: :
0assignvariableop_2_quantize_layer_optimizer_step: 8
.assignvariableop_3_quant_conv2d_optimizer_step: 8
*assignvariableop_4_quant_conv2d_kernel_min: 8
*assignvariableop_5_quant_conv2d_kernel_max: =
3assignvariableop_6_quant_conv2d_post_activation_min: =
3assignvariableop_7_quant_conv2d_post_activation_max: ?
5assignvariableop_8_quant_max_pooling2d_optimizer_step: :
0assignvariableop_9_quant_conv2d_1_optimizer_step: ;
-assignvariableop_10_quant_conv2d_1_kernel_min: ;
-assignvariableop_11_quant_conv2d_1_kernel_max: @
6assignvariableop_12_quant_conv2d_1_post_activation_min: @
6assignvariableop_13_quant_conv2d_1_post_activation_max: B
8assignvariableop_14_quant_max_pooling2d_1_optimizer_step: ;
1assignvariableop_15_quant_conv2d_2_optimizer_step: ;
-assignvariableop_16_quant_conv2d_2_kernel_min:@;
-assignvariableop_17_quant_conv2d_2_kernel_max:@@
6assignvariableop_18_quant_conv2d_2_post_activation_min: @
6assignvariableop_19_quant_conv2d_2_post_activation_max: ;
1assignvariableop_20_quant_conv2d_3_optimizer_step: ;
-assignvariableop_21_quant_conv2d_3_kernel_min:@;
-assignvariableop_22_quant_conv2d_3_kernel_max:@@
6assignvariableop_23_quant_conv2d_3_post_activation_min: @
6assignvariableop_24_quant_conv2d_3_post_activation_max: ;
1assignvariableop_25_quant_conv2d_4_optimizer_step: ;
-assignvariableop_26_quant_conv2d_4_kernel_min: ;
-assignvariableop_27_quant_conv2d_4_kernel_max: @
6assignvariableop_28_quant_conv2d_4_post_activation_min: @
6assignvariableop_29_quant_conv2d_4_post_activation_max: B
8assignvariableop_30_quant_max_pooling2d_2_optimizer_step: :
0assignvariableop_31_quant_flatten_optimizer_step: 8
.assignvariableop_32_quant_dense_optimizer_step: 4
*assignvariableop_33_quant_dense_kernel_min: 4
*assignvariableop_34_quant_dense_kernel_max: =
3assignvariableop_35_quant_dense_post_activation_min: =
3assignvariableop_36_quant_dense_post_activation_max: :
0assignvariableop_37_quant_dropout_optimizer_step: :
0assignvariableop_38_quant_dense_1_optimizer_step: 6
,assignvariableop_39_quant_dense_1_kernel_min: 6
,assignvariableop_40_quant_dense_1_kernel_max: ?
5assignvariableop_41_quant_dense_1_post_activation_min: ?
5assignvariableop_42_quant_dense_1_post_activation_max: <
2assignvariableop_43_quant_dropout_1_optimizer_step: :
0assignvariableop_44_quant_dense_2_optimizer_step: 6
,assignvariableop_45_quant_dense_2_kernel_min: 6
,assignvariableop_46_quant_dense_2_kernel_max: >
4assignvariableop_47_quant_dense_2_pre_activation_min: >
4assignvariableop_48_quant_dense_2_pre_activation_max: ;
!assignvariableop_49_conv2d_kernel: -
assignvariableop_50_conv2d_bias: =
#assignvariableop_51_conv2d_1_kernel:  /
!assignvariableop_52_conv2d_1_bias: =
#assignvariableop_53_conv2d_2_kernel: @/
!assignvariableop_54_conv2d_2_bias:@=
#assignvariableop_55_conv2d_3_kernel:@@/
!assignvariableop_56_conv2d_3_bias:@=
#assignvariableop_57_conv2d_4_kernel:@ /
!assignvariableop_58_conv2d_4_bias: 4
 assignvariableop_59_dense_kernel:
-
assignvariableop_60_dense_bias:	6
"assignvariableop_61_dense_1_kernel:
/
 assignvariableop_62_dense_1_bias:	5
"assignvariableop_63_dense_2_kernel:	.
 assignvariableop_64_dense_2_bias:'
assignvariableop_65_adam_iter:	 )
assignvariableop_66_adam_beta_1: )
assignvariableop_67_adam_beta_2: (
assignvariableop_68_adam_decay: 0
&assignvariableop_69_adam_learning_rate: %
assignvariableop_70_total_1: %
assignvariableop_71_count_1: #
assignvariableop_72_total: #
assignvariableop_73_count: B
(assignvariableop_74_adam_conv2d_kernel_m: 4
&assignvariableop_75_adam_conv2d_bias_m: D
*assignvariableop_76_adam_conv2d_1_kernel_m:  6
(assignvariableop_77_adam_conv2d_1_bias_m: D
*assignvariableop_78_adam_conv2d_2_kernel_m: @6
(assignvariableop_79_adam_conv2d_2_bias_m:@D
*assignvariableop_80_adam_conv2d_3_kernel_m:@@6
(assignvariableop_81_adam_conv2d_3_bias_m:@D
*assignvariableop_82_adam_conv2d_4_kernel_m:@ 6
(assignvariableop_83_adam_conv2d_4_bias_m: ;
'assignvariableop_84_adam_dense_kernel_m:
4
%assignvariableop_85_adam_dense_bias_m:	=
)assignvariableop_86_adam_dense_1_kernel_m:
6
'assignvariableop_87_adam_dense_1_bias_m:	<
)assignvariableop_88_adam_dense_2_kernel_m:	5
'assignvariableop_89_adam_dense_2_bias_m:B
(assignvariableop_90_adam_conv2d_kernel_v: 4
&assignvariableop_91_adam_conv2d_bias_v: D
*assignvariableop_92_adam_conv2d_1_kernel_v:  6
(assignvariableop_93_adam_conv2d_1_bias_v: D
*assignvariableop_94_adam_conv2d_2_kernel_v: @6
(assignvariableop_95_adam_conv2d_2_bias_v:@D
*assignvariableop_96_adam_conv2d_3_kernel_v:@@6
(assignvariableop_97_adam_conv2d_3_bias_v:@D
*assignvariableop_98_adam_conv2d_4_kernel_v:@ 6
(assignvariableop_99_adam_conv2d_4_bias_v: <
(assignvariableop_100_adam_dense_kernel_v:
5
&assignvariableop_101_adam_dense_bias_v:	>
*assignvariableop_102_adam_dense_1_kernel_v:
7
(assignvariableop_103_adam_dense_1_bias_v:	=
*assignvariableop_104_adam_dense_2_kernel_v:	6
(assignvariableop_105_adam_dense_2_bias_v:
identity_107¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_100¢AssignVariableOp_101¢AssignVariableOp_102¢AssignVariableOp_103¢AssignVariableOp_104¢AssignVariableOp_105¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_28¢AssignVariableOp_29¢AssignVariableOp_3¢AssignVariableOp_30¢AssignVariableOp_31¢AssignVariableOp_32¢AssignVariableOp_33¢AssignVariableOp_34¢AssignVariableOp_35¢AssignVariableOp_36¢AssignVariableOp_37¢AssignVariableOp_38¢AssignVariableOp_39¢AssignVariableOp_4¢AssignVariableOp_40¢AssignVariableOp_41¢AssignVariableOp_42¢AssignVariableOp_43¢AssignVariableOp_44¢AssignVariableOp_45¢AssignVariableOp_46¢AssignVariableOp_47¢AssignVariableOp_48¢AssignVariableOp_49¢AssignVariableOp_5¢AssignVariableOp_50¢AssignVariableOp_51¢AssignVariableOp_52¢AssignVariableOp_53¢AssignVariableOp_54¢AssignVariableOp_55¢AssignVariableOp_56¢AssignVariableOp_57¢AssignVariableOp_58¢AssignVariableOp_59¢AssignVariableOp_6¢AssignVariableOp_60¢AssignVariableOp_61¢AssignVariableOp_62¢AssignVariableOp_63¢AssignVariableOp_64¢AssignVariableOp_65¢AssignVariableOp_66¢AssignVariableOp_67¢AssignVariableOp_68¢AssignVariableOp_69¢AssignVariableOp_7¢AssignVariableOp_70¢AssignVariableOp_71¢AssignVariableOp_72¢AssignVariableOp_73¢AssignVariableOp_74¢AssignVariableOp_75¢AssignVariableOp_76¢AssignVariableOp_77¢AssignVariableOp_78¢AssignVariableOp_79¢AssignVariableOp_8¢AssignVariableOp_80¢AssignVariableOp_81¢AssignVariableOp_82¢AssignVariableOp_83¢AssignVariableOp_84¢AssignVariableOp_85¢AssignVariableOp_86¢AssignVariableOp_87¢AssignVariableOp_88¢AssignVariableOp_89¢AssignVariableOp_9¢AssignVariableOp_90¢AssignVariableOp_91¢AssignVariableOp_92¢AssignVariableOp_93¢AssignVariableOp_94¢AssignVariableOp_95¢AssignVariableOp_96¢AssignVariableOp_97¢AssignVariableOp_98¢AssignVariableOp_99Û3
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:k*
dtype0*3
value÷2Bô2kBBlayer_with_weights-0/quantize_layer_min/.ATTRIBUTES/VARIABLE_VALUEBBlayer_with_weights-0/quantize_layer_max/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-0/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-1/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-1/kernel_min/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-1/kernel_max/.ATTRIBUTES/VARIABLE_VALUEBClayer_with_weights-1/post_activation_min/.ATTRIBUTES/VARIABLE_VALUEBClayer_with_weights-1/post_activation_max/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-2/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-3/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-3/kernel_min/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-3/kernel_max/.ATTRIBUTES/VARIABLE_VALUEBClayer_with_weights-3/post_activation_min/.ATTRIBUTES/VARIABLE_VALUEBClayer_with_weights-3/post_activation_max/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-4/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-5/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-5/kernel_min/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-5/kernel_max/.ATTRIBUTES/VARIABLE_VALUEBClayer_with_weights-5/post_activation_min/.ATTRIBUTES/VARIABLE_VALUEBClayer_with_weights-5/post_activation_max/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-6/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-6/kernel_min/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-6/kernel_max/.ATTRIBUTES/VARIABLE_VALUEBClayer_with_weights-6/post_activation_min/.ATTRIBUTES/VARIABLE_VALUEBClayer_with_weights-6/post_activation_max/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-7/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-7/kernel_min/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-7/kernel_max/.ATTRIBUTES/VARIABLE_VALUEBClayer_with_weights-7/post_activation_min/.ATTRIBUTES/VARIABLE_VALUEBClayer_with_weights-7/post_activation_max/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-8/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-9/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-10/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-10/kernel_min/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-10/kernel_max/.ATTRIBUTES/VARIABLE_VALUEBDlayer_with_weights-10/post_activation_min/.ATTRIBUTES/VARIABLE_VALUEBDlayer_with_weights-10/post_activation_max/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-11/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-12/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-12/kernel_min/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-12/kernel_max/.ATTRIBUTES/VARIABLE_VALUEBDlayer_with_weights-12/post_activation_min/.ATTRIBUTES/VARIABLE_VALUEBDlayer_with_weights-12/post_activation_max/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-13/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-14/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-14/kernel_min/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-14/kernel_max/.ATTRIBUTES/VARIABLE_VALUEBClayer_with_weights-14/pre_activation_min/.ATTRIBUTES/VARIABLE_VALUEBClayer_with_weights-14/pre_activation_max/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/26/.ATTRIBUTES/VARIABLE_VALUEB'variables/27/.ATTRIBUTES/VARIABLE_VALUEB'variables/33/.ATTRIBUTES/VARIABLE_VALUEB'variables/34/.ATTRIBUTES/VARIABLE_VALUEB'variables/42/.ATTRIBUTES/VARIABLE_VALUEB'variables/43/.ATTRIBUTES/VARIABLE_VALUEB'variables/50/.ATTRIBUTES/VARIABLE_VALUEB'variables/51/.ATTRIBUTES/VARIABLE_VALUEB'variables/58/.ATTRIBUTES/VARIABLE_VALUEB'variables/59/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/19/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/20/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/26/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/27/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/33/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/34/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/42/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/43/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/50/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/51/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/58/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/59/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/19/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/20/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/26/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/27/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/33/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/34/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/42/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/43/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/50/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/51/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/58/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/59/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHÉ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:k*
dtype0*ë
valueáBÞkB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ¸
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*Â
_output_shapes¯
¬:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*y
dtypeso
m2k	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOpAssignVariableOp2assignvariableop_quantize_layer_quantize_layer_minIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:£
AssignVariableOp_1AssignVariableOp4assignvariableop_1_quantize_layer_quantize_layer_maxIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_2AssignVariableOp0assignvariableop_2_quantize_layer_optimizer_stepIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_3AssignVariableOp.assignvariableop_3_quant_conv2d_optimizer_stepIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_4AssignVariableOp*assignvariableop_4_quant_conv2d_kernel_minIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_5AssignVariableOp*assignvariableop_5_quant_conv2d_kernel_maxIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:¢
AssignVariableOp_6AssignVariableOp3assignvariableop_6_quant_conv2d_post_activation_minIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:¢
AssignVariableOp_7AssignVariableOp3assignvariableop_7_quant_conv2d_post_activation_maxIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:¤
AssignVariableOp_8AssignVariableOp5assignvariableop_8_quant_max_pooling2d_optimizer_stepIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_9AssignVariableOp0assignvariableop_9_quant_conv2d_1_optimizer_stepIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_10AssignVariableOp-assignvariableop_10_quant_conv2d_1_kernel_minIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_11AssignVariableOp-assignvariableop_11_quant_conv2d_1_kernel_maxIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_12AssignVariableOp6assignvariableop_12_quant_conv2d_1_post_activation_minIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_13AssignVariableOp6assignvariableop_13_quant_conv2d_1_post_activation_maxIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_14AssignVariableOp8assignvariableop_14_quant_max_pooling2d_1_optimizer_stepIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:¢
AssignVariableOp_15AssignVariableOp1assignvariableop_15_quant_conv2d_2_optimizer_stepIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_16AssignVariableOp-assignvariableop_16_quant_conv2d_2_kernel_minIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_17AssignVariableOp-assignvariableop_17_quant_conv2d_2_kernel_maxIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_18AssignVariableOp6assignvariableop_18_quant_conv2d_2_post_activation_minIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_19AssignVariableOp6assignvariableop_19_quant_conv2d_2_post_activation_maxIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:¢
AssignVariableOp_20AssignVariableOp1assignvariableop_20_quant_conv2d_3_optimizer_stepIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_21AssignVariableOp-assignvariableop_21_quant_conv2d_3_kernel_minIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_22AssignVariableOp-assignvariableop_22_quant_conv2d_3_kernel_maxIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_23AssignVariableOp6assignvariableop_23_quant_conv2d_3_post_activation_minIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_24AssignVariableOp6assignvariableop_24_quant_conv2d_3_post_activation_maxIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:¢
AssignVariableOp_25AssignVariableOp1assignvariableop_25_quant_conv2d_4_optimizer_stepIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_26AssignVariableOp-assignvariableop_26_quant_conv2d_4_kernel_minIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_27AssignVariableOp-assignvariableop_27_quant_conv2d_4_kernel_maxIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_28AssignVariableOp6assignvariableop_28_quant_conv2d_4_post_activation_minIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_29AssignVariableOp6assignvariableop_29_quant_conv2d_4_post_activation_maxIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_30AssignVariableOp8assignvariableop_30_quant_max_pooling2d_2_optimizer_stepIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:¡
AssignVariableOp_31AssignVariableOp0assignvariableop_31_quant_flatten_optimizer_stepIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_32AssignVariableOp.assignvariableop_32_quant_dense_optimizer_stepIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_33AssignVariableOp*assignvariableop_33_quant_dense_kernel_minIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_34AssignVariableOp*assignvariableop_34_quant_dense_kernel_maxIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:¤
AssignVariableOp_35AssignVariableOp3assignvariableop_35_quant_dense_post_activation_minIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:¤
AssignVariableOp_36AssignVariableOp3assignvariableop_36_quant_dense_post_activation_maxIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:¡
AssignVariableOp_37AssignVariableOp0assignvariableop_37_quant_dropout_optimizer_stepIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:¡
AssignVariableOp_38AssignVariableOp0assignvariableop_38_quant_dense_1_optimizer_stepIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_39AssignVariableOp,assignvariableop_39_quant_dense_1_kernel_minIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_40AssignVariableOp,assignvariableop_40_quant_dense_1_kernel_maxIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:¦
AssignVariableOp_41AssignVariableOp5assignvariableop_41_quant_dense_1_post_activation_minIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:¦
AssignVariableOp_42AssignVariableOp5assignvariableop_42_quant_dense_1_post_activation_maxIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:£
AssignVariableOp_43AssignVariableOp2assignvariableop_43_quant_dropout_1_optimizer_stepIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:¡
AssignVariableOp_44AssignVariableOp0assignvariableop_44_quant_dense_2_optimizer_stepIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_45AssignVariableOp,assignvariableop_45_quant_dense_2_kernel_minIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_46AssignVariableOp,assignvariableop_46_quant_dense_2_kernel_maxIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:¥
AssignVariableOp_47AssignVariableOp4assignvariableop_47_quant_dense_2_pre_activation_minIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:¥
AssignVariableOp_48AssignVariableOp4assignvariableop_48_quant_dense_2_pre_activation_maxIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_49AssignVariableOp!assignvariableop_49_conv2d_kernelIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_50AssignVariableOpassignvariableop_50_conv2d_biasIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_51AssignVariableOp#assignvariableop_51_conv2d_1_kernelIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_52AssignVariableOp!assignvariableop_52_conv2d_1_biasIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_53AssignVariableOp#assignvariableop_53_conv2d_2_kernelIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_54AssignVariableOp!assignvariableop_54_conv2d_2_biasIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_55AssignVariableOp#assignvariableop_55_conv2d_3_kernelIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_56AssignVariableOp!assignvariableop_56_conv2d_3_biasIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_57AssignVariableOp#assignvariableop_57_conv2d_4_kernelIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_58AssignVariableOp!assignvariableop_58_conv2d_4_biasIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_59AssignVariableOp assignvariableop_59_dense_kernelIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_60AssignVariableOpassignvariableop_60_dense_biasIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_61AssignVariableOp"assignvariableop_61_dense_1_kernelIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_62AssignVariableOp assignvariableop_62_dense_1_biasIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_63AssignVariableOp"assignvariableop_63_dense_2_kernelIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_64AssignVariableOp assignvariableop_64_dense_2_biasIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0	*
_output_shapes
:
AssignVariableOp_65AssignVariableOpassignvariableop_65_adam_iterIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_66AssignVariableOpassignvariableop_66_adam_beta_1Identity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_67AssignVariableOpassignvariableop_67_adam_beta_2Identity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_68AssignVariableOpassignvariableop_68_adam_decayIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_69AssignVariableOp&assignvariableop_69_adam_learning_rateIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_70AssignVariableOpassignvariableop_70_total_1Identity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_71AssignVariableOpassignvariableop_71_count_1Identity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_72AssignVariableOpassignvariableop_72_totalIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_73AssignVariableOpassignvariableop_73_countIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_74AssignVariableOp(assignvariableop_74_adam_conv2d_kernel_mIdentity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_75AssignVariableOp&assignvariableop_75_adam_conv2d_bias_mIdentity_75:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_76AssignVariableOp*assignvariableop_76_adam_conv2d_1_kernel_mIdentity_76:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_77AssignVariableOp(assignvariableop_77_adam_conv2d_1_bias_mIdentity_77:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_78AssignVariableOp*assignvariableop_78_adam_conv2d_2_kernel_mIdentity_78:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_79IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_79AssignVariableOp(assignvariableop_79_adam_conv2d_2_bias_mIdentity_79:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_80IdentityRestoreV2:tensors:80"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_80AssignVariableOp*assignvariableop_80_adam_conv2d_3_kernel_mIdentity_80:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_81IdentityRestoreV2:tensors:81"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_81AssignVariableOp(assignvariableop_81_adam_conv2d_3_bias_mIdentity_81:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_82IdentityRestoreV2:tensors:82"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_82AssignVariableOp*assignvariableop_82_adam_conv2d_4_kernel_mIdentity_82:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_83IdentityRestoreV2:tensors:83"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_83AssignVariableOp(assignvariableop_83_adam_conv2d_4_bias_mIdentity_83:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_84IdentityRestoreV2:tensors:84"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_84AssignVariableOp'assignvariableop_84_adam_dense_kernel_mIdentity_84:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_85IdentityRestoreV2:tensors:85"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_85AssignVariableOp%assignvariableop_85_adam_dense_bias_mIdentity_85:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_86IdentityRestoreV2:tensors:86"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_86AssignVariableOp)assignvariableop_86_adam_dense_1_kernel_mIdentity_86:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_87IdentityRestoreV2:tensors:87"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_87AssignVariableOp'assignvariableop_87_adam_dense_1_bias_mIdentity_87:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_88IdentityRestoreV2:tensors:88"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_88AssignVariableOp)assignvariableop_88_adam_dense_2_kernel_mIdentity_88:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_89IdentityRestoreV2:tensors:89"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_89AssignVariableOp'assignvariableop_89_adam_dense_2_bias_mIdentity_89:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_90IdentityRestoreV2:tensors:90"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_90AssignVariableOp(assignvariableop_90_adam_conv2d_kernel_vIdentity_90:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_91IdentityRestoreV2:tensors:91"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_91AssignVariableOp&assignvariableop_91_adam_conv2d_bias_vIdentity_91:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_92IdentityRestoreV2:tensors:92"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_92AssignVariableOp*assignvariableop_92_adam_conv2d_1_kernel_vIdentity_92:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_93IdentityRestoreV2:tensors:93"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_93AssignVariableOp(assignvariableop_93_adam_conv2d_1_bias_vIdentity_93:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_94IdentityRestoreV2:tensors:94"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_94AssignVariableOp*assignvariableop_94_adam_conv2d_2_kernel_vIdentity_94:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_95IdentityRestoreV2:tensors:95"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_95AssignVariableOp(assignvariableop_95_adam_conv2d_2_bias_vIdentity_95:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_96IdentityRestoreV2:tensors:96"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_96AssignVariableOp*assignvariableop_96_adam_conv2d_3_kernel_vIdentity_96:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_97IdentityRestoreV2:tensors:97"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_97AssignVariableOp(assignvariableop_97_adam_conv2d_3_bias_vIdentity_97:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_98IdentityRestoreV2:tensors:98"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_98AssignVariableOp*assignvariableop_98_adam_conv2d_4_kernel_vIdentity_98:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_99IdentityRestoreV2:tensors:99"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_99AssignVariableOp(assignvariableop_99_adam_conv2d_4_bias_vIdentity_99:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_100IdentityRestoreV2:tensors:100"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_100AssignVariableOp(assignvariableop_100_adam_dense_kernel_vIdentity_100:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_101IdentityRestoreV2:tensors:101"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_101AssignVariableOp&assignvariableop_101_adam_dense_bias_vIdentity_101:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_102IdentityRestoreV2:tensors:102"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_102AssignVariableOp*assignvariableop_102_adam_dense_1_kernel_vIdentity_102:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_103IdentityRestoreV2:tensors:103"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_103AssignVariableOp(assignvariableop_103_adam_dense_1_bias_vIdentity_103:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_104IdentityRestoreV2:tensors:104"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_104AssignVariableOp*assignvariableop_104_adam_dense_2_kernel_vIdentity_104:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_105IdentityRestoreV2:tensors:105"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_105AssignVariableOp(assignvariableop_105_adam_dense_2_bias_vIdentity_105:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ò
Identity_106Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_103^AssignVariableOp_104^AssignVariableOp_105^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99^NoOp"/device:CPU:0*
T0*
_output_shapes
: Y
Identity_107IdentityIdentity_106:output:0^NoOp_1*
T0*
_output_shapes
: Þ
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_103^AssignVariableOp_104^AssignVariableOp_105^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99*"
_acd_function_control_output(*
_output_shapes
 "%
identity_107Identity_107:output:0*ë
_input_shapesÙ
Ö: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102,
AssignVariableOp_100AssignVariableOp_1002,
AssignVariableOp_101AssignVariableOp_1012,
AssignVariableOp_102AssignVariableOp_1022,
AssignVariableOp_103AssignVariableOp_1032,
AssignVariableOp_104AssignVariableOp_1042,
AssignVariableOp_105AssignVariableOp_1052*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_7AssignVariableOp_72*
AssignVariableOp_70AssignVariableOp_702*
AssignVariableOp_71AssignVariableOp_712*
AssignVariableOp_72AssignVariableOp_722*
AssignVariableOp_73AssignVariableOp_732*
AssignVariableOp_74AssignVariableOp_742*
AssignVariableOp_75AssignVariableOp_752*
AssignVariableOp_76AssignVariableOp_762*
AssignVariableOp_77AssignVariableOp_772*
AssignVariableOp_78AssignVariableOp_782*
AssignVariableOp_79AssignVariableOp_792(
AssignVariableOp_8AssignVariableOp_82*
AssignVariableOp_80AssignVariableOp_802*
AssignVariableOp_81AssignVariableOp_812*
AssignVariableOp_82AssignVariableOp_822*
AssignVariableOp_83AssignVariableOp_832*
AssignVariableOp_84AssignVariableOp_842*
AssignVariableOp_85AssignVariableOp_852*
AssignVariableOp_86AssignVariableOp_862*
AssignVariableOp_87AssignVariableOp_872*
AssignVariableOp_88AssignVariableOp_882*
AssignVariableOp_89AssignVariableOp_892(
AssignVariableOp_9AssignVariableOp_92*
AssignVariableOp_90AssignVariableOp_902*
AssignVariableOp_91AssignVariableOp_912*
AssignVariableOp_92AssignVariableOp_922*
AssignVariableOp_93AssignVariableOp_932*
AssignVariableOp_94AssignVariableOp_942*
AssignVariableOp_95AssignVariableOp_952*
AssignVariableOp_96AssignVariableOp_962*
AssignVariableOp_97AssignVariableOp_972*
AssignVariableOp_98AssignVariableOp_982*
AssignVariableOp_99AssignVariableOp_99:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
­X
º	
J__inference_quant_conv2d_1_layer_call_and_return_conditional_losses_144131

inputsI
/lastvaluequant_batchmin_readvariableop_resource:  3
%lastvaluequant_assignminlast_resource: 3
%lastvaluequant_assignmaxlast_resource: -
biasadd_readvariableop_resource: @
6movingavgquantize_assignminema_readvariableop_resource: @
6movingavgquantize_assignmaxema_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢LastValueQuant/AssignMaxLast¢LastValueQuant/AssignMinLast¢&LastValueQuant/BatchMax/ReadVariableOp¢&LastValueQuant/BatchMin/ReadVariableOp¢?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp¢ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1¢ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2¢2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp¢-MovingAvgQuantize/AssignMaxEma/ReadVariableOp¢2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp¢-MovingAvgQuantize/AssignMinEma/ReadVariableOp¢8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp¢:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1¢1conv2d_1/kernel/Regularizer/L2Loss/ReadVariableOp
&LastValueQuant/BatchMin/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:  *
dtype0~
)LastValueQuant/BatchMin/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          §
LastValueQuant/BatchMinMin.LastValueQuant/BatchMin/ReadVariableOp:value:02LastValueQuant/BatchMin/reduction_indices:output:0*
T0*
_output_shapes
: 
&LastValueQuant/BatchMax/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:  *
dtype0~
)LastValueQuant/BatchMax/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          §
LastValueQuant/BatchMaxMax.LastValueQuant/BatchMax/ReadVariableOp:value:02LastValueQuant/BatchMax/reduction_indices:output:0*
T0*
_output_shapes
: ]
LastValueQuant/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ¿
LastValueQuant/truedivRealDiv LastValueQuant/BatchMax:output:0!LastValueQuant/truediv/y:output:0*
T0*
_output_shapes
: 
LastValueQuant/MinimumMinimum LastValueQuant/BatchMin:output:0LastValueQuant/truediv:z:0*
T0*
_output_shapes
: Y
LastValueQuant/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ¿
LastValueQuant/mulMul LastValueQuant/BatchMin:output:0LastValueQuant/mul/y:output:0*
T0*
_output_shapes
: 
LastValueQuant/MaximumMaximum LastValueQuant/BatchMax:output:0LastValueQuant/mul:z:0*
T0*
_output_shapes
: ¬
LastValueQuant/AssignMinLastAssignVariableOp%lastvaluequant_assignminlast_resourceLastValueQuant/Minimum:z:0*
_output_shapes
 *
dtype0*
validate_shape(¬
LastValueQuant/AssignMaxLastAssignVariableOp%lastvaluequant_assignmaxlast_resourceLastValueQuant/Maximum:z:0*
_output_shapes
 *
dtype0*
validate_shape(·
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:  *
dtype0Â
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ReadVariableOp%lastvaluequant_assignminlast_resource^LastValueQuant/AssignMinLast*
_output_shapes
: *
dtype0Â
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ReadVariableOp%lastvaluequant_assignmaxlast_resource^LastValueQuant/AssignMaxLast*
_output_shapes
: *
dtype0ð
0LastValueQuant/FakeQuantWithMinMaxVarsPerChannel!FakeQuantWithMinMaxVarsPerChannelGLastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2:value:0*&
_output_shapes
:  *
narrow_range(¶
Conv2DConv2Dinputs:LastValueQuant/FakeQuantWithMinMaxVarsPerChannel:outputs:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿKK *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿKK X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿKK p
MovingAvgQuantize/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             x
MovingAvgQuantize/BatchMinMinRelu:activations:0 MovingAvgQuantize/Const:output:0*
T0*
_output_shapes
: r
MovingAvgQuantize/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             z
MovingAvgQuantize/BatchMaxMaxRelu:activations:0"MovingAvgQuantize/Const_1:output:0*
T0*
_output_shapes
: `
MovingAvgQuantize/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
MovingAvgQuantize/MinimumMinimum#MovingAvgQuantize/BatchMin:output:0$MovingAvgQuantize/Minimum/y:output:0*
T0*
_output_shapes
: `
MovingAvgQuantize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
MovingAvgQuantize/MaximumMaximum#MovingAvgQuantize/BatchMax:output:0$MovingAvgQuantize/Maximum/y:output:0*
T0*
_output_shapes
: i
$MovingAvgQuantize/AssignMinEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o:
-MovingAvgQuantize/AssignMinEma/ReadVariableOpReadVariableOp6movingavgquantize_assignminema_readvariableop_resource*
_output_shapes
: *
dtype0 
"MovingAvgQuantize/AssignMinEma/subSub5MovingAvgQuantize/AssignMinEma/ReadVariableOp:value:0MovingAvgQuantize/Minimum:z:0*
T0*
_output_shapes
: ¡
"MovingAvgQuantize/AssignMinEma/mulMul&MovingAvgQuantize/AssignMinEma/sub:z:0-MovingAvgQuantize/AssignMinEma/decay:output:0*
T0*
_output_shapes
: ü
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOpAssignSubVariableOp6movingavgquantize_assignminema_readvariableop_resource&MovingAvgQuantize/AssignMinEma/mul:z:0.^MovingAvgQuantize/AssignMinEma/ReadVariableOp*
_output_shapes
 *
dtype0i
$MovingAvgQuantize/AssignMaxEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o:
-MovingAvgQuantize/AssignMaxEma/ReadVariableOpReadVariableOp6movingavgquantize_assignmaxema_readvariableop_resource*
_output_shapes
: *
dtype0 
"MovingAvgQuantize/AssignMaxEma/subSub5MovingAvgQuantize/AssignMaxEma/ReadVariableOp:value:0MovingAvgQuantize/Maximum:z:0*
T0*
_output_shapes
: ¡
"MovingAvgQuantize/AssignMaxEma/mulMul&MovingAvgQuantize/AssignMaxEma/sub:z:0-MovingAvgQuantize/AssignMaxEma/decay:output:0*
T0*
_output_shapes
: ü
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpAssignSubVariableOp6movingavgquantize_assignmaxema_readvariableop_resource&MovingAvgQuantize/AssignMaxEma/mul:z:0.^MovingAvgQuantize/AssignMaxEma/ReadVariableOp*
_output_shapes
 *
dtype0Ü
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp6movingavgquantize_assignminema_readvariableop_resource3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp*
_output_shapes
: *
dtype0Þ
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp6movingavgquantize_assignmaxema_readvariableop_resource3^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp*
_output_shapes
: *
dtype0
)MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsRelu:activations:0@MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BMovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿKK ©
1conv2d_1/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:  *
dtype0
"conv2d_1/kernel/Regularizer/L2LossL2Loss9conv2d_1/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!conv2d_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o; 
conv2d_1/kernel/Regularizer/mulMul*conv2d_1/kernel/Regularizer/mul/x:output:0+conv2d_1/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: 
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿKK ¯
NoOpNoOp^BiasAdd/ReadVariableOp^LastValueQuant/AssignMaxLast^LastValueQuant/AssignMinLast'^LastValueQuant/BatchMax/ReadVariableOp'^LastValueQuant/BatchMin/ReadVariableOp@^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpB^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1B^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_23^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp.^MovingAvgQuantize/AssignMaxEma/ReadVariableOp3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp.^MovingAvgQuantize/AssignMinEma/ReadVariableOp9^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12^conv2d_1/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿKK : : : : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2<
LastValueQuant/AssignMaxLastLastValueQuant/AssignMaxLast2<
LastValueQuant/AssignMinLastLastValueQuant/AssignMinLast2P
&LastValueQuant/BatchMax/ReadVariableOp&LastValueQuant/BatchMax/ReadVariableOp2P
&LastValueQuant/BatchMin/ReadVariableOp&LastValueQuant/BatchMin/ReadVariableOp2
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp2
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_12
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_22h
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2^
-MovingAvgQuantize/AssignMaxEma/ReadVariableOp-MovingAvgQuantize/AssignMaxEma/ReadVariableOp2h
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2^
-MovingAvgQuantize/AssignMinEma/ReadVariableOp-MovingAvgQuantize/AssignMinEma/ReadVariableOp2t
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12f
1conv2d_1/kernel/Regularizer/L2Loss/ReadVariableOp1conv2d_1/kernel/Regularizer/L2Loss/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿKK 
 
_user_specified_nameinputs
å
R
6__inference_quant_max_pooling2d_1_layer_call_fn_146340

inputs
identityÒ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *;
config_proto+)

CPU

GPU2*0J
	
   E 8 *Z
fURS
Q__inference_quant_max_pooling2d_1_layer_call_and_return_conditional_losses_143046h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿKK :W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿKK 
 
_user_specified_nameinputs
­X
º	
J__inference_quant_conv2d_4_layer_call_and_return_conditional_losses_143851

inputsI
/lastvaluequant_batchmin_readvariableop_resource:@ 3
%lastvaluequant_assignminlast_resource: 3
%lastvaluequant_assignmaxlast_resource: -
biasadd_readvariableop_resource: @
6movingavgquantize_assignminema_readvariableop_resource: @
6movingavgquantize_assignmaxema_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢LastValueQuant/AssignMaxLast¢LastValueQuant/AssignMinLast¢&LastValueQuant/BatchMax/ReadVariableOp¢&LastValueQuant/BatchMin/ReadVariableOp¢?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp¢ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1¢ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2¢2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp¢-MovingAvgQuantize/AssignMaxEma/ReadVariableOp¢2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp¢-MovingAvgQuantize/AssignMinEma/ReadVariableOp¢8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp¢:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1¢1conv2d_4/kernel/Regularizer/L2Loss/ReadVariableOp
&LastValueQuant/BatchMin/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:@ *
dtype0~
)LastValueQuant/BatchMin/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          §
LastValueQuant/BatchMinMin.LastValueQuant/BatchMin/ReadVariableOp:value:02LastValueQuant/BatchMin/reduction_indices:output:0*
T0*
_output_shapes
: 
&LastValueQuant/BatchMax/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:@ *
dtype0~
)LastValueQuant/BatchMax/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          §
LastValueQuant/BatchMaxMax.LastValueQuant/BatchMax/ReadVariableOp:value:02LastValueQuant/BatchMax/reduction_indices:output:0*
T0*
_output_shapes
: ]
LastValueQuant/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ¿
LastValueQuant/truedivRealDiv LastValueQuant/BatchMax:output:0!LastValueQuant/truediv/y:output:0*
T0*
_output_shapes
: 
LastValueQuant/MinimumMinimum LastValueQuant/BatchMin:output:0LastValueQuant/truediv:z:0*
T0*
_output_shapes
: Y
LastValueQuant/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ¿
LastValueQuant/mulMul LastValueQuant/BatchMin:output:0LastValueQuant/mul/y:output:0*
T0*
_output_shapes
: 
LastValueQuant/MaximumMaximum LastValueQuant/BatchMax:output:0LastValueQuant/mul:z:0*
T0*
_output_shapes
: ¬
LastValueQuant/AssignMinLastAssignVariableOp%lastvaluequant_assignminlast_resourceLastValueQuant/Minimum:z:0*
_output_shapes
 *
dtype0*
validate_shape(¬
LastValueQuant/AssignMaxLastAssignVariableOp%lastvaluequant_assignmaxlast_resourceLastValueQuant/Maximum:z:0*
_output_shapes
 *
dtype0*
validate_shape(·
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:@ *
dtype0Â
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ReadVariableOp%lastvaluequant_assignminlast_resource^LastValueQuant/AssignMinLast*
_output_shapes
: *
dtype0Â
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ReadVariableOp%lastvaluequant_assignmaxlast_resource^LastValueQuant/AssignMaxLast*
_output_shapes
: *
dtype0ð
0LastValueQuant/FakeQuantWithMinMaxVarsPerChannel!FakeQuantWithMinMaxVarsPerChannelGLastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2:value:0*&
_output_shapes
:@ *
narrow_range(¶
Conv2DConv2Dinputs:LastValueQuant/FakeQuantWithMinMaxVarsPerChannel:outputs:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ p
MovingAvgQuantize/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             x
MovingAvgQuantize/BatchMinMinRelu:activations:0 MovingAvgQuantize/Const:output:0*
T0*
_output_shapes
: r
MovingAvgQuantize/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             z
MovingAvgQuantize/BatchMaxMaxRelu:activations:0"MovingAvgQuantize/Const_1:output:0*
T0*
_output_shapes
: `
MovingAvgQuantize/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
MovingAvgQuantize/MinimumMinimum#MovingAvgQuantize/BatchMin:output:0$MovingAvgQuantize/Minimum/y:output:0*
T0*
_output_shapes
: `
MovingAvgQuantize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
MovingAvgQuantize/MaximumMaximum#MovingAvgQuantize/BatchMax:output:0$MovingAvgQuantize/Maximum/y:output:0*
T0*
_output_shapes
: i
$MovingAvgQuantize/AssignMinEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o:
-MovingAvgQuantize/AssignMinEma/ReadVariableOpReadVariableOp6movingavgquantize_assignminema_readvariableop_resource*
_output_shapes
: *
dtype0 
"MovingAvgQuantize/AssignMinEma/subSub5MovingAvgQuantize/AssignMinEma/ReadVariableOp:value:0MovingAvgQuantize/Minimum:z:0*
T0*
_output_shapes
: ¡
"MovingAvgQuantize/AssignMinEma/mulMul&MovingAvgQuantize/AssignMinEma/sub:z:0-MovingAvgQuantize/AssignMinEma/decay:output:0*
T0*
_output_shapes
: ü
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOpAssignSubVariableOp6movingavgquantize_assignminema_readvariableop_resource&MovingAvgQuantize/AssignMinEma/mul:z:0.^MovingAvgQuantize/AssignMinEma/ReadVariableOp*
_output_shapes
 *
dtype0i
$MovingAvgQuantize/AssignMaxEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o:
-MovingAvgQuantize/AssignMaxEma/ReadVariableOpReadVariableOp6movingavgquantize_assignmaxema_readvariableop_resource*
_output_shapes
: *
dtype0 
"MovingAvgQuantize/AssignMaxEma/subSub5MovingAvgQuantize/AssignMaxEma/ReadVariableOp:value:0MovingAvgQuantize/Maximum:z:0*
T0*
_output_shapes
: ¡
"MovingAvgQuantize/AssignMaxEma/mulMul&MovingAvgQuantize/AssignMaxEma/sub:z:0-MovingAvgQuantize/AssignMaxEma/decay:output:0*
T0*
_output_shapes
: ü
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpAssignSubVariableOp6movingavgquantize_assignmaxema_readvariableop_resource&MovingAvgQuantize/AssignMaxEma/mul:z:0.^MovingAvgQuantize/AssignMaxEma/ReadVariableOp*
_output_shapes
 *
dtype0Ü
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp6movingavgquantize_assignminema_readvariableop_resource3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp*
_output_shapes
: *
dtype0Þ
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp6movingavgquantize_assignmaxema_readvariableop_resource3^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp*
_output_shapes
: *
dtype0
)MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsRelu:activations:0@MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BMovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ©
1conv2d_4/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:@ *
dtype0
"conv2d_4/kernel/Regularizer/L2LossL2Loss9conv2d_4/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!conv2d_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o; 
conv2d_4/kernel/Regularizer/mulMul*conv2d_4/kernel/Regularizer/mul/x:output:0+conv2d_4/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: 
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¯
NoOpNoOp^BiasAdd/ReadVariableOp^LastValueQuant/AssignMaxLast^LastValueQuant/AssignMinLast'^LastValueQuant/BatchMax/ReadVariableOp'^LastValueQuant/BatchMin/ReadVariableOp@^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpB^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1B^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_23^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp.^MovingAvgQuantize/AssignMaxEma/ReadVariableOp3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp.^MovingAvgQuantize/AssignMinEma/ReadVariableOp9^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12^conv2d_4/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ@: : : : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2<
LastValueQuant/AssignMaxLastLastValueQuant/AssignMaxLast2<
LastValueQuant/AssignMinLastLastValueQuant/AssignMinLast2P
&LastValueQuant/BatchMax/ReadVariableOp&LastValueQuant/BatchMax/ReadVariableOp2P
&LastValueQuant/BatchMin/ReadVariableOp&LastValueQuant/BatchMin/ReadVariableOp2
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp2
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_12
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_22h
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2^
-MovingAvgQuantize/AssignMaxEma/ReadVariableOp-MovingAvgQuantize/AssignMaxEma/ReadVariableOp2h
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2^
-MovingAvgQuantize/AssignMinEma/ReadVariableOp-MovingAvgQuantize/AssignMinEma/ReadVariableOp2t
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12f
1conv2d_4/kernel/Regularizer/L2Loss/ReadVariableOp1conv2d_4/kernel/Regularizer/L2Loss/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Á&

G__inference_quant_dense_layer_call_and_return_conditional_losses_143205

inputsR
>lastvaluequant_fakequantwithminmaxvars_readvariableop_resource:
J
@lastvaluequant_fakequantwithminmaxvars_readvariableop_1_resource: J
@lastvaluequant_fakequantwithminmaxvars_readvariableop_2_resource: .
biasadd_readvariableop_resource:	K
Amovingavgquantize_fakequantwithminmaxvars_readvariableop_resource: M
Cmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: 
identity¢BiasAdd/ReadVariableOp¢5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp¢7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1¢7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2¢8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp¢:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1¢.dense/kernel/Regularizer/L2Loss/ReadVariableOp¶
5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp>lastvaluequant_fakequantwithminmaxvars_readvariableop_resource* 
_output_shapes
:
*
dtype0°
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp@lastvaluequant_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0°
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2ReadVariableOp@lastvaluequant_fakequantwithminmaxvars_readvariableop_2_resource*
_output_shapes
: *
dtype0¸
&LastValueQuant/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars=LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp:value:0?LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0?LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2:value:0* 
_output_shapes
:
*
narrow_range(}
MatMulMatMulinputs0LastValueQuant/FakeQuantWithMinMaxVars:outputs:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ²
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpAmovingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype0¶
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpCmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0
)MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsRelu:activations:0@MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BMovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¯
.dense/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp>lastvaluequant_fakequantwithminmaxvars_readvariableop_resource* 
_output_shapes
:
*
dtype0
dense/kernel/Regularizer/L2LossL2Loss6dense/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: c
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0(dense/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: 
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´
NoOpNoOp^BiasAdd/ReadVariableOp6^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp8^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_18^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_29^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1/^dense/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ: : : : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2n
5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp2r
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_17LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_12r
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_27LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_22t
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12`
.dense/kernel/Regularizer/L2Loss/ReadVariableOp.dense/kernel/Regularizer/L2Loss/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
í
Z
F__inference_sequential_layer_call_and_return_conditional_losses_146035

inputsJ
@quantize_layer_allvaluesquantize_minimum_readvariableop_resource: J
@quantize_layer_allvaluesquantize_maximum_readvariableop_resource: V
<quant_conv2d_lastvaluequant_batchmin_readvariableop_resource: @
2quant_conv2d_lastvaluequant_assignminlast_resource: @
2quant_conv2d_lastvaluequant_assignmaxlast_resource: :
,quant_conv2d_biasadd_readvariableop_resource: M
Cquant_conv2d_movingavgquantize_assignminema_readvariableop_resource: M
Cquant_conv2d_movingavgquantize_assignmaxema_readvariableop_resource: X
>quant_conv2d_1_lastvaluequant_batchmin_readvariableop_resource:  B
4quant_conv2d_1_lastvaluequant_assignminlast_resource: B
4quant_conv2d_1_lastvaluequant_assignmaxlast_resource: <
.quant_conv2d_1_biasadd_readvariableop_resource: O
Equant_conv2d_1_movingavgquantize_assignminema_readvariableop_resource: O
Equant_conv2d_1_movingavgquantize_assignmaxema_readvariableop_resource: X
>quant_conv2d_2_lastvaluequant_batchmin_readvariableop_resource: @B
4quant_conv2d_2_lastvaluequant_assignminlast_resource:@B
4quant_conv2d_2_lastvaluequant_assignmaxlast_resource:@<
.quant_conv2d_2_biasadd_readvariableop_resource:@O
Equant_conv2d_2_movingavgquantize_assignminema_readvariableop_resource: O
Equant_conv2d_2_movingavgquantize_assignmaxema_readvariableop_resource: X
>quant_conv2d_3_lastvaluequant_batchmin_readvariableop_resource:@@B
4quant_conv2d_3_lastvaluequant_assignminlast_resource:@B
4quant_conv2d_3_lastvaluequant_assignmaxlast_resource:@<
.quant_conv2d_3_biasadd_readvariableop_resource:@O
Equant_conv2d_3_movingavgquantize_assignminema_readvariableop_resource: O
Equant_conv2d_3_movingavgquantize_assignmaxema_readvariableop_resource: X
>quant_conv2d_4_lastvaluequant_batchmin_readvariableop_resource:@ B
4quant_conv2d_4_lastvaluequant_assignminlast_resource: B
4quant_conv2d_4_lastvaluequant_assignmaxlast_resource: <
.quant_conv2d_4_biasadd_readvariableop_resource: O
Equant_conv2d_4_movingavgquantize_assignminema_readvariableop_resource: O
Equant_conv2d_4_movingavgquantize_assignmaxema_readvariableop_resource: O
;quant_dense_lastvaluequant_batchmin_readvariableop_resource:
;
1quant_dense_lastvaluequant_assignminlast_resource: ;
1quant_dense_lastvaluequant_assignmaxlast_resource: :
+quant_dense_biasadd_readvariableop_resource:	L
Bquant_dense_movingavgquantize_assignminema_readvariableop_resource: L
Bquant_dense_movingavgquantize_assignmaxema_readvariableop_resource: Q
=quant_dense_1_lastvaluequant_batchmin_readvariableop_resource:
=
3quant_dense_1_lastvaluequant_assignminlast_resource: =
3quant_dense_1_lastvaluequant_assignmaxlast_resource: <
-quant_dense_1_biasadd_readvariableop_resource:	N
Dquant_dense_1_movingavgquantize_assignminema_readvariableop_resource: N
Dquant_dense_1_movingavgquantize_assignmaxema_readvariableop_resource: P
=quant_dense_2_lastvaluequant_batchmin_readvariableop_resource:	=
3quant_dense_2_lastvaluequant_assignminlast_resource: =
3quant_dense_2_lastvaluequant_assignmaxlast_resource: ;
-quant_dense_2_biasadd_readvariableop_resource:N
Dquant_dense_2_movingavgquantize_assignminema_readvariableop_resource: N
Dquant_dense_2_movingavgquantize_assignmaxema_readvariableop_resource: 
identity¢/conv2d/kernel/Regularizer/L2Loss/ReadVariableOp¢1conv2d_1/kernel/Regularizer/L2Loss/ReadVariableOp¢1conv2d_2/kernel/Regularizer/L2Loss/ReadVariableOp¢1conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOp¢1conv2d_4/kernel/Regularizer/L2Loss/ReadVariableOp¢.dense/kernel/Regularizer/L2Loss/ReadVariableOp¢0dense_1/kernel/Regularizer/L2Loss/ReadVariableOp¢#quant_conv2d/BiasAdd/ReadVariableOp¢)quant_conv2d/LastValueQuant/AssignMaxLast¢)quant_conv2d/LastValueQuant/AssignMinLast¢3quant_conv2d/LastValueQuant/BatchMax/ReadVariableOp¢3quant_conv2d/LastValueQuant/BatchMin/ReadVariableOp¢Lquant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp¢Nquant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1¢Nquant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2¢?quant_conv2d/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp¢:quant_conv2d/MovingAvgQuantize/AssignMaxEma/ReadVariableOp¢?quant_conv2d/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp¢:quant_conv2d/MovingAvgQuantize/AssignMinEma/ReadVariableOp¢Equant_conv2d/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp¢Gquant_conv2d/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1¢%quant_conv2d_1/BiasAdd/ReadVariableOp¢+quant_conv2d_1/LastValueQuant/AssignMaxLast¢+quant_conv2d_1/LastValueQuant/AssignMinLast¢5quant_conv2d_1/LastValueQuant/BatchMax/ReadVariableOp¢5quant_conv2d_1/LastValueQuant/BatchMin/ReadVariableOp¢Nquant_conv2d_1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp¢Pquant_conv2d_1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1¢Pquant_conv2d_1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2¢Aquant_conv2d_1/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp¢<quant_conv2d_1/MovingAvgQuantize/AssignMaxEma/ReadVariableOp¢Aquant_conv2d_1/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp¢<quant_conv2d_1/MovingAvgQuantize/AssignMinEma/ReadVariableOp¢Gquant_conv2d_1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp¢Iquant_conv2d_1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1¢%quant_conv2d_2/BiasAdd/ReadVariableOp¢+quant_conv2d_2/LastValueQuant/AssignMaxLast¢+quant_conv2d_2/LastValueQuant/AssignMinLast¢5quant_conv2d_2/LastValueQuant/BatchMax/ReadVariableOp¢5quant_conv2d_2/LastValueQuant/BatchMin/ReadVariableOp¢Nquant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp¢Pquant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1¢Pquant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2¢Aquant_conv2d_2/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp¢<quant_conv2d_2/MovingAvgQuantize/AssignMaxEma/ReadVariableOp¢Aquant_conv2d_2/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp¢<quant_conv2d_2/MovingAvgQuantize/AssignMinEma/ReadVariableOp¢Gquant_conv2d_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp¢Iquant_conv2d_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1¢%quant_conv2d_3/BiasAdd/ReadVariableOp¢+quant_conv2d_3/LastValueQuant/AssignMaxLast¢+quant_conv2d_3/LastValueQuant/AssignMinLast¢5quant_conv2d_3/LastValueQuant/BatchMax/ReadVariableOp¢5quant_conv2d_3/LastValueQuant/BatchMin/ReadVariableOp¢Nquant_conv2d_3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp¢Pquant_conv2d_3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1¢Pquant_conv2d_3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2¢Aquant_conv2d_3/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp¢<quant_conv2d_3/MovingAvgQuantize/AssignMaxEma/ReadVariableOp¢Aquant_conv2d_3/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp¢<quant_conv2d_3/MovingAvgQuantize/AssignMinEma/ReadVariableOp¢Gquant_conv2d_3/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp¢Iquant_conv2d_3/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1¢%quant_conv2d_4/BiasAdd/ReadVariableOp¢+quant_conv2d_4/LastValueQuant/AssignMaxLast¢+quant_conv2d_4/LastValueQuant/AssignMinLast¢5quant_conv2d_4/LastValueQuant/BatchMax/ReadVariableOp¢5quant_conv2d_4/LastValueQuant/BatchMin/ReadVariableOp¢Nquant_conv2d_4/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp¢Pquant_conv2d_4/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1¢Pquant_conv2d_4/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2¢Aquant_conv2d_4/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp¢<quant_conv2d_4/MovingAvgQuantize/AssignMaxEma/ReadVariableOp¢Aquant_conv2d_4/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp¢<quant_conv2d_4/MovingAvgQuantize/AssignMinEma/ReadVariableOp¢Gquant_conv2d_4/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp¢Iquant_conv2d_4/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1¢"quant_dense/BiasAdd/ReadVariableOp¢(quant_dense/LastValueQuant/AssignMaxLast¢(quant_dense/LastValueQuant/AssignMinLast¢2quant_dense/LastValueQuant/BatchMax/ReadVariableOp¢2quant_dense/LastValueQuant/BatchMin/ReadVariableOp¢Aquant_dense/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp¢Cquant_dense/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1¢Cquant_dense/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2¢>quant_dense/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp¢9quant_dense/MovingAvgQuantize/AssignMaxEma/ReadVariableOp¢>quant_dense/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp¢9quant_dense/MovingAvgQuantize/AssignMinEma/ReadVariableOp¢Dquant_dense/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp¢Fquant_dense/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1¢$quant_dense_1/BiasAdd/ReadVariableOp¢*quant_dense_1/LastValueQuant/AssignMaxLast¢*quant_dense_1/LastValueQuant/AssignMinLast¢4quant_dense_1/LastValueQuant/BatchMax/ReadVariableOp¢4quant_dense_1/LastValueQuant/BatchMin/ReadVariableOp¢Cquant_dense_1/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp¢Equant_dense_1/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1¢Equant_dense_1/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2¢@quant_dense_1/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp¢;quant_dense_1/MovingAvgQuantize/AssignMaxEma/ReadVariableOp¢@quant_dense_1/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp¢;quant_dense_1/MovingAvgQuantize/AssignMinEma/ReadVariableOp¢Fquant_dense_1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp¢Hquant_dense_1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1¢$quant_dense_2/BiasAdd/ReadVariableOp¢*quant_dense_2/LastValueQuant/AssignMaxLast¢*quant_dense_2/LastValueQuant/AssignMinLast¢4quant_dense_2/LastValueQuant/BatchMax/ReadVariableOp¢4quant_dense_2/LastValueQuant/BatchMin/ReadVariableOp¢Cquant_dense_2/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp¢Equant_dense_2/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1¢Equant_dense_2/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2¢@quant_dense_2/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp¢;quant_dense_2/MovingAvgQuantize/AssignMaxEma/ReadVariableOp¢@quant_dense_2/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp¢;quant_dense_2/MovingAvgQuantize/AssignMinEma/ReadVariableOp¢Fquant_dense_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp¢Hquant_dense_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1¢2quantize_layer/AllValuesQuantize/AssignMaxAllValue¢2quantize_layer/AllValuesQuantize/AssignMinAllValue¢Gquantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp¢Iquantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1¢7quantize_layer/AllValuesQuantize/Maximum/ReadVariableOp¢7quantize_layer/AllValuesQuantize/Minimum/ReadVariableOp
&quantize_layer/AllValuesQuantize/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
)quantize_layer/AllValuesQuantize/BatchMinMininputs/quantize_layer/AllValuesQuantize/Const:output:0*
T0*
_output_shapes
: 
(quantize_layer/AllValuesQuantize/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             
)quantize_layer/AllValuesQuantize/BatchMaxMaxinputs1quantize_layer/AllValuesQuantize/Const_1:output:0*
T0*
_output_shapes
: °
7quantize_layer/AllValuesQuantize/Minimum/ReadVariableOpReadVariableOp@quantize_layer_allvaluesquantize_minimum_readvariableop_resource*
_output_shapes
: *
dtype0É
(quantize_layer/AllValuesQuantize/MinimumMinimum?quantize_layer/AllValuesQuantize/Minimum/ReadVariableOp:value:02quantize_layer/AllValuesQuantize/BatchMin:output:0*
T0*
_output_shapes
: q
,quantize_layer/AllValuesQuantize/Minimum_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    »
*quantize_layer/AllValuesQuantize/Minimum_1Minimum,quantize_layer/AllValuesQuantize/Minimum:z:05quantize_layer/AllValuesQuantize/Minimum_1/y:output:0*
T0*
_output_shapes
: °
7quantize_layer/AllValuesQuantize/Maximum/ReadVariableOpReadVariableOp@quantize_layer_allvaluesquantize_maximum_readvariableop_resource*
_output_shapes
: *
dtype0É
(quantize_layer/AllValuesQuantize/MaximumMaximum?quantize_layer/AllValuesQuantize/Maximum/ReadVariableOp:value:02quantize_layer/AllValuesQuantize/BatchMax:output:0*
T0*
_output_shapes
: q
,quantize_layer/AllValuesQuantize/Maximum_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    »
*quantize_layer/AllValuesQuantize/Maximum_1Maximum,quantize_layer/AllValuesQuantize/Maximum:z:05quantize_layer/AllValuesQuantize/Maximum_1/y:output:0*
T0*
_output_shapes
: «
2quantize_layer/AllValuesQuantize/AssignMinAllValueAssignVariableOp@quantize_layer_allvaluesquantize_minimum_readvariableop_resource.quantize_layer/AllValuesQuantize/Minimum_1:z:08^quantize_layer/AllValuesQuantize/Minimum/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(«
2quantize_layer/AllValuesQuantize/AssignMaxAllValueAssignVariableOp@quantize_layer_allvaluesquantize_maximum_readvariableop_resource.quantize_layer/AllValuesQuantize/Maximum_1:z:08^quantize_layer/AllValuesQuantize/Maximum/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(õ
Gquantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp@quantize_layer_allvaluesquantize_minimum_readvariableop_resource3^quantize_layer/AllValuesQuantize/AssignMinAllValue*
_output_shapes
: *
dtype0÷
Iquantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp@quantize_layer_allvaluesquantize_maximum_readvariableop_resource3^quantize_layer/AllValuesQuantize/AssignMaxAllValue*
_output_shapes
: *
dtype0²
8quantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsinputsOquantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Qquantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿææ¸
3quant_conv2d/LastValueQuant/BatchMin/ReadVariableOpReadVariableOp<quant_conv2d_lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
: *
dtype0
6quant_conv2d/LastValueQuant/BatchMin/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          Î
$quant_conv2d/LastValueQuant/BatchMinMin;quant_conv2d/LastValueQuant/BatchMin/ReadVariableOp:value:0?quant_conv2d/LastValueQuant/BatchMin/reduction_indices:output:0*
T0*
_output_shapes
: ¸
3quant_conv2d/LastValueQuant/BatchMax/ReadVariableOpReadVariableOp<quant_conv2d_lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
: *
dtype0
6quant_conv2d/LastValueQuant/BatchMax/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          Î
$quant_conv2d/LastValueQuant/BatchMaxMax;quant_conv2d/LastValueQuant/BatchMax/ReadVariableOp:value:0?quant_conv2d/LastValueQuant/BatchMax/reduction_indices:output:0*
T0*
_output_shapes
: j
%quant_conv2d/LastValueQuant/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ¿²
#quant_conv2d/LastValueQuant/truedivRealDiv-quant_conv2d/LastValueQuant/BatchMax:output:0.quant_conv2d/LastValueQuant/truediv/y:output:0*
T0*
_output_shapes
: «
#quant_conv2d/LastValueQuant/MinimumMinimum-quant_conv2d/LastValueQuant/BatchMin:output:0'quant_conv2d/LastValueQuant/truediv:z:0*
T0*
_output_shapes
: f
!quant_conv2d/LastValueQuant/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ¿¦
quant_conv2d/LastValueQuant/mulMul-quant_conv2d/LastValueQuant/BatchMin:output:0*quant_conv2d/LastValueQuant/mul/y:output:0*
T0*
_output_shapes
: §
#quant_conv2d/LastValueQuant/MaximumMaximum-quant_conv2d/LastValueQuant/BatchMax:output:0#quant_conv2d/LastValueQuant/mul:z:0*
T0*
_output_shapes
: Ó
)quant_conv2d/LastValueQuant/AssignMinLastAssignVariableOp2quant_conv2d_lastvaluequant_assignminlast_resource'quant_conv2d/LastValueQuant/Minimum:z:0*
_output_shapes
 *
dtype0*
validate_shape(Ó
)quant_conv2d/LastValueQuant/AssignMaxLastAssignVariableOp2quant_conv2d_lastvaluequant_assignmaxlast_resource'quant_conv2d/LastValueQuant/Maximum:z:0*
_output_shapes
 *
dtype0*
validate_shape(Ñ
Lquant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpReadVariableOp<quant_conv2d_lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
: *
dtype0é
Nquant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ReadVariableOp2quant_conv2d_lastvaluequant_assignminlast_resource*^quant_conv2d/LastValueQuant/AssignMinLast*
_output_shapes
: *
dtype0é
Nquant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ReadVariableOp2quant_conv2d_lastvaluequant_assignmaxlast_resource*^quant_conv2d/LastValueQuant/AssignMaxLast*
_output_shapes
: *
dtype0¤
=quant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel!FakeQuantWithMinMaxVarsPerChannelTquant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp:value:0Vquant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1:value:0Vquant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2:value:0*&
_output_shapes
: *
narrow_range(
quant_conv2d/Conv2DConv2DBquantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars:outputs:0Gquant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel:outputs:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿââ *
paddingVALID*
strides

#quant_conv2d/BiasAdd/ReadVariableOpReadVariableOp,quant_conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0¦
quant_conv2d/BiasAddBiasAddquant_conv2d/Conv2D:output:0+quant_conv2d/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿââ t
quant_conv2d/ReluReluquant_conv2d/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿââ }
$quant_conv2d/MovingAvgQuantize/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
'quant_conv2d/MovingAvgQuantize/BatchMinMinquant_conv2d/Relu:activations:0-quant_conv2d/MovingAvgQuantize/Const:output:0*
T0*
_output_shapes
: 
&quant_conv2d/MovingAvgQuantize/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             ¡
'quant_conv2d/MovingAvgQuantize/BatchMaxMaxquant_conv2d/Relu:activations:0/quant_conv2d/MovingAvgQuantize/Const_1:output:0*
T0*
_output_shapes
: m
(quant_conv2d/MovingAvgQuantize/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ·
&quant_conv2d/MovingAvgQuantize/MinimumMinimum0quant_conv2d/MovingAvgQuantize/BatchMin:output:01quant_conv2d/MovingAvgQuantize/Minimum/y:output:0*
T0*
_output_shapes
: m
(quant_conv2d/MovingAvgQuantize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ·
&quant_conv2d/MovingAvgQuantize/MaximumMaximum0quant_conv2d/MovingAvgQuantize/BatchMax:output:01quant_conv2d/MovingAvgQuantize/Maximum/y:output:0*
T0*
_output_shapes
: v
1quant_conv2d/MovingAvgQuantize/AssignMinEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o:¶
:quant_conv2d/MovingAvgQuantize/AssignMinEma/ReadVariableOpReadVariableOpCquant_conv2d_movingavgquantize_assignminema_readvariableop_resource*
_output_shapes
: *
dtype0Ç
/quant_conv2d/MovingAvgQuantize/AssignMinEma/subSubBquant_conv2d/MovingAvgQuantize/AssignMinEma/ReadVariableOp:value:0*quant_conv2d/MovingAvgQuantize/Minimum:z:0*
T0*
_output_shapes
: È
/quant_conv2d/MovingAvgQuantize/AssignMinEma/mulMul3quant_conv2d/MovingAvgQuantize/AssignMinEma/sub:z:0:quant_conv2d/MovingAvgQuantize/AssignMinEma/decay:output:0*
T0*
_output_shapes
: °
?quant_conv2d/MovingAvgQuantize/AssignMinEma/AssignSubVariableOpAssignSubVariableOpCquant_conv2d_movingavgquantize_assignminema_readvariableop_resource3quant_conv2d/MovingAvgQuantize/AssignMinEma/mul:z:0;^quant_conv2d/MovingAvgQuantize/AssignMinEma/ReadVariableOp*
_output_shapes
 *
dtype0v
1quant_conv2d/MovingAvgQuantize/AssignMaxEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o:¶
:quant_conv2d/MovingAvgQuantize/AssignMaxEma/ReadVariableOpReadVariableOpCquant_conv2d_movingavgquantize_assignmaxema_readvariableop_resource*
_output_shapes
: *
dtype0Ç
/quant_conv2d/MovingAvgQuantize/AssignMaxEma/subSubBquant_conv2d/MovingAvgQuantize/AssignMaxEma/ReadVariableOp:value:0*quant_conv2d/MovingAvgQuantize/Maximum:z:0*
T0*
_output_shapes
: È
/quant_conv2d/MovingAvgQuantize/AssignMaxEma/mulMul3quant_conv2d/MovingAvgQuantize/AssignMaxEma/sub:z:0:quant_conv2d/MovingAvgQuantize/AssignMaxEma/decay:output:0*
T0*
_output_shapes
: °
?quant_conv2d/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpAssignSubVariableOpCquant_conv2d_movingavgquantize_assignmaxema_readvariableop_resource3quant_conv2d/MovingAvgQuantize/AssignMaxEma/mul:z:0;^quant_conv2d/MovingAvgQuantize/AssignMaxEma/ReadVariableOp*
_output_shapes
 *
dtype0
Equant_conv2d/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpCquant_conv2d_movingavgquantize_assignminema_readvariableop_resource@^quant_conv2d/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp*
_output_shapes
: *
dtype0
Gquant_conv2d/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpCquant_conv2d_movingavgquantize_assignmaxema_readvariableop_resource@^quant_conv2d/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp*
_output_shapes
: *
dtype0Å
6quant_conv2d/MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsquant_conv2d/Relu:activations:0Mquant_conv2d/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Oquant_conv2d/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿââ Õ
quant_max_pooling2d/MaxPoolMaxPool@quant_conv2d/MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿKK *
ksize
*
paddingVALID*
strides
¼
5quant_conv2d_1/LastValueQuant/BatchMin/ReadVariableOpReadVariableOp>quant_conv2d_1_lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:  *
dtype0
8quant_conv2d_1/LastValueQuant/BatchMin/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          Ô
&quant_conv2d_1/LastValueQuant/BatchMinMin=quant_conv2d_1/LastValueQuant/BatchMin/ReadVariableOp:value:0Aquant_conv2d_1/LastValueQuant/BatchMin/reduction_indices:output:0*
T0*
_output_shapes
: ¼
5quant_conv2d_1/LastValueQuant/BatchMax/ReadVariableOpReadVariableOp>quant_conv2d_1_lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:  *
dtype0
8quant_conv2d_1/LastValueQuant/BatchMax/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          Ô
&quant_conv2d_1/LastValueQuant/BatchMaxMax=quant_conv2d_1/LastValueQuant/BatchMax/ReadVariableOp:value:0Aquant_conv2d_1/LastValueQuant/BatchMax/reduction_indices:output:0*
T0*
_output_shapes
: l
'quant_conv2d_1/LastValueQuant/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ¿¸
%quant_conv2d_1/LastValueQuant/truedivRealDiv/quant_conv2d_1/LastValueQuant/BatchMax:output:00quant_conv2d_1/LastValueQuant/truediv/y:output:0*
T0*
_output_shapes
: ±
%quant_conv2d_1/LastValueQuant/MinimumMinimum/quant_conv2d_1/LastValueQuant/BatchMin:output:0)quant_conv2d_1/LastValueQuant/truediv:z:0*
T0*
_output_shapes
: h
#quant_conv2d_1/LastValueQuant/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ¿¬
!quant_conv2d_1/LastValueQuant/mulMul/quant_conv2d_1/LastValueQuant/BatchMin:output:0,quant_conv2d_1/LastValueQuant/mul/y:output:0*
T0*
_output_shapes
: ­
%quant_conv2d_1/LastValueQuant/MaximumMaximum/quant_conv2d_1/LastValueQuant/BatchMax:output:0%quant_conv2d_1/LastValueQuant/mul:z:0*
T0*
_output_shapes
: Ù
+quant_conv2d_1/LastValueQuant/AssignMinLastAssignVariableOp4quant_conv2d_1_lastvaluequant_assignminlast_resource)quant_conv2d_1/LastValueQuant/Minimum:z:0*
_output_shapes
 *
dtype0*
validate_shape(Ù
+quant_conv2d_1/LastValueQuant/AssignMaxLastAssignVariableOp4quant_conv2d_1_lastvaluequant_assignmaxlast_resource)quant_conv2d_1/LastValueQuant/Maximum:z:0*
_output_shapes
 *
dtype0*
validate_shape(Õ
Nquant_conv2d_1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpReadVariableOp>quant_conv2d_1_lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:  *
dtype0ï
Pquant_conv2d_1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ReadVariableOp4quant_conv2d_1_lastvaluequant_assignminlast_resource,^quant_conv2d_1/LastValueQuant/AssignMinLast*
_output_shapes
: *
dtype0ï
Pquant_conv2d_1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ReadVariableOp4quant_conv2d_1_lastvaluequant_assignmaxlast_resource,^quant_conv2d_1/LastValueQuant/AssignMaxLast*
_output_shapes
: *
dtype0¬
?quant_conv2d_1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel!FakeQuantWithMinMaxVarsPerChannelVquant_conv2d_1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp:value:0Xquant_conv2d_1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1:value:0Xquant_conv2d_1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2:value:0*&
_output_shapes
:  *
narrow_range(ò
quant_conv2d_1/Conv2DConv2D$quant_max_pooling2d/MaxPool:output:0Iquant_conv2d_1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel:outputs:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿKK *
paddingSAME*
strides

%quant_conv2d_1/BiasAdd/ReadVariableOpReadVariableOp.quant_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0ª
quant_conv2d_1/BiasAddBiasAddquant_conv2d_1/Conv2D:output:0-quant_conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿKK v
quant_conv2d_1/ReluReluquant_conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿKK 
&quant_conv2d_1/MovingAvgQuantize/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             ¥
)quant_conv2d_1/MovingAvgQuantize/BatchMinMin!quant_conv2d_1/Relu:activations:0/quant_conv2d_1/MovingAvgQuantize/Const:output:0*
T0*
_output_shapes
: 
(quant_conv2d_1/MovingAvgQuantize/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             §
)quant_conv2d_1/MovingAvgQuantize/BatchMaxMax!quant_conv2d_1/Relu:activations:01quant_conv2d_1/MovingAvgQuantize/Const_1:output:0*
T0*
_output_shapes
: o
*quant_conv2d_1/MovingAvgQuantize/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ½
(quant_conv2d_1/MovingAvgQuantize/MinimumMinimum2quant_conv2d_1/MovingAvgQuantize/BatchMin:output:03quant_conv2d_1/MovingAvgQuantize/Minimum/y:output:0*
T0*
_output_shapes
: o
*quant_conv2d_1/MovingAvgQuantize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ½
(quant_conv2d_1/MovingAvgQuantize/MaximumMaximum2quant_conv2d_1/MovingAvgQuantize/BatchMax:output:03quant_conv2d_1/MovingAvgQuantize/Maximum/y:output:0*
T0*
_output_shapes
: x
3quant_conv2d_1/MovingAvgQuantize/AssignMinEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o:º
<quant_conv2d_1/MovingAvgQuantize/AssignMinEma/ReadVariableOpReadVariableOpEquant_conv2d_1_movingavgquantize_assignminema_readvariableop_resource*
_output_shapes
: *
dtype0Í
1quant_conv2d_1/MovingAvgQuantize/AssignMinEma/subSubDquant_conv2d_1/MovingAvgQuantize/AssignMinEma/ReadVariableOp:value:0,quant_conv2d_1/MovingAvgQuantize/Minimum:z:0*
T0*
_output_shapes
: Î
1quant_conv2d_1/MovingAvgQuantize/AssignMinEma/mulMul5quant_conv2d_1/MovingAvgQuantize/AssignMinEma/sub:z:0<quant_conv2d_1/MovingAvgQuantize/AssignMinEma/decay:output:0*
T0*
_output_shapes
: ¸
Aquant_conv2d_1/MovingAvgQuantize/AssignMinEma/AssignSubVariableOpAssignSubVariableOpEquant_conv2d_1_movingavgquantize_assignminema_readvariableop_resource5quant_conv2d_1/MovingAvgQuantize/AssignMinEma/mul:z:0=^quant_conv2d_1/MovingAvgQuantize/AssignMinEma/ReadVariableOp*
_output_shapes
 *
dtype0x
3quant_conv2d_1/MovingAvgQuantize/AssignMaxEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o:º
<quant_conv2d_1/MovingAvgQuantize/AssignMaxEma/ReadVariableOpReadVariableOpEquant_conv2d_1_movingavgquantize_assignmaxema_readvariableop_resource*
_output_shapes
: *
dtype0Í
1quant_conv2d_1/MovingAvgQuantize/AssignMaxEma/subSubDquant_conv2d_1/MovingAvgQuantize/AssignMaxEma/ReadVariableOp:value:0,quant_conv2d_1/MovingAvgQuantize/Maximum:z:0*
T0*
_output_shapes
: Î
1quant_conv2d_1/MovingAvgQuantize/AssignMaxEma/mulMul5quant_conv2d_1/MovingAvgQuantize/AssignMaxEma/sub:z:0<quant_conv2d_1/MovingAvgQuantize/AssignMaxEma/decay:output:0*
T0*
_output_shapes
: ¸
Aquant_conv2d_1/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpAssignSubVariableOpEquant_conv2d_1_movingavgquantize_assignmaxema_readvariableop_resource5quant_conv2d_1/MovingAvgQuantize/AssignMaxEma/mul:z:0=^quant_conv2d_1/MovingAvgQuantize/AssignMaxEma/ReadVariableOp*
_output_shapes
 *
dtype0
Gquant_conv2d_1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpEquant_conv2d_1_movingavgquantize_assignminema_readvariableop_resourceB^quant_conv2d_1/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp*
_output_shapes
: *
dtype0
Iquant_conv2d_1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpEquant_conv2d_1_movingavgquantize_assignmaxema_readvariableop_resourceB^quant_conv2d_1/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp*
_output_shapes
: *
dtype0Ë
8quant_conv2d_1/MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars!quant_conv2d_1/Relu:activations:0Oquant_conv2d_1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Qquant_conv2d_1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿKK Ù
quant_max_pooling2d_1/MaxPoolMaxPoolBquant_conv2d_1/MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
ksize
*
paddingVALID*
strides
¼
5quant_conv2d_2/LastValueQuant/BatchMin/ReadVariableOpReadVariableOp>quant_conv2d_2_lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
: @*
dtype0
8quant_conv2d_2/LastValueQuant/BatchMin/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          Ô
&quant_conv2d_2/LastValueQuant/BatchMinMin=quant_conv2d_2/LastValueQuant/BatchMin/ReadVariableOp:value:0Aquant_conv2d_2/LastValueQuant/BatchMin/reduction_indices:output:0*
T0*
_output_shapes
:@¼
5quant_conv2d_2/LastValueQuant/BatchMax/ReadVariableOpReadVariableOp>quant_conv2d_2_lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
: @*
dtype0
8quant_conv2d_2/LastValueQuant/BatchMax/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          Ô
&quant_conv2d_2/LastValueQuant/BatchMaxMax=quant_conv2d_2/LastValueQuant/BatchMax/ReadVariableOp:value:0Aquant_conv2d_2/LastValueQuant/BatchMax/reduction_indices:output:0*
T0*
_output_shapes
:@l
'quant_conv2d_2/LastValueQuant/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ¿¸
%quant_conv2d_2/LastValueQuant/truedivRealDiv/quant_conv2d_2/LastValueQuant/BatchMax:output:00quant_conv2d_2/LastValueQuant/truediv/y:output:0*
T0*
_output_shapes
:@±
%quant_conv2d_2/LastValueQuant/MinimumMinimum/quant_conv2d_2/LastValueQuant/BatchMin:output:0)quant_conv2d_2/LastValueQuant/truediv:z:0*
T0*
_output_shapes
:@h
#quant_conv2d_2/LastValueQuant/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ¿¬
!quant_conv2d_2/LastValueQuant/mulMul/quant_conv2d_2/LastValueQuant/BatchMin:output:0,quant_conv2d_2/LastValueQuant/mul/y:output:0*
T0*
_output_shapes
:@­
%quant_conv2d_2/LastValueQuant/MaximumMaximum/quant_conv2d_2/LastValueQuant/BatchMax:output:0%quant_conv2d_2/LastValueQuant/mul:z:0*
T0*
_output_shapes
:@Ù
+quant_conv2d_2/LastValueQuant/AssignMinLastAssignVariableOp4quant_conv2d_2_lastvaluequant_assignminlast_resource)quant_conv2d_2/LastValueQuant/Minimum:z:0*
_output_shapes
 *
dtype0*
validate_shape(Ù
+quant_conv2d_2/LastValueQuant/AssignMaxLastAssignVariableOp4quant_conv2d_2_lastvaluequant_assignmaxlast_resource)quant_conv2d_2/LastValueQuant/Maximum:z:0*
_output_shapes
 *
dtype0*
validate_shape(Õ
Nquant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpReadVariableOp>quant_conv2d_2_lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
: @*
dtype0ï
Pquant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ReadVariableOp4quant_conv2d_2_lastvaluequant_assignminlast_resource,^quant_conv2d_2/LastValueQuant/AssignMinLast*
_output_shapes
:@*
dtype0ï
Pquant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ReadVariableOp4quant_conv2d_2_lastvaluequant_assignmaxlast_resource,^quant_conv2d_2/LastValueQuant/AssignMaxLast*
_output_shapes
:@*
dtype0¬
?quant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel!FakeQuantWithMinMaxVarsPerChannelVquant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp:value:0Xquant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1:value:0Xquant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2:value:0*&
_output_shapes
: @*
narrow_range(ô
quant_conv2d_2/Conv2DConv2D&quant_max_pooling2d_1/MaxPool:output:0Iquant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel:outputs:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingSAME*
strides

%quant_conv2d_2/BiasAdd/ReadVariableOpReadVariableOp.quant_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0ª
quant_conv2d_2/BiasAddBiasAddquant_conv2d_2/Conv2D:output:0-quant_conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@v
quant_conv2d_2/ReluReluquant_conv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
&quant_conv2d_2/MovingAvgQuantize/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             ¥
)quant_conv2d_2/MovingAvgQuantize/BatchMinMin!quant_conv2d_2/Relu:activations:0/quant_conv2d_2/MovingAvgQuantize/Const:output:0*
T0*
_output_shapes
: 
(quant_conv2d_2/MovingAvgQuantize/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             §
)quant_conv2d_2/MovingAvgQuantize/BatchMaxMax!quant_conv2d_2/Relu:activations:01quant_conv2d_2/MovingAvgQuantize/Const_1:output:0*
T0*
_output_shapes
: o
*quant_conv2d_2/MovingAvgQuantize/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ½
(quant_conv2d_2/MovingAvgQuantize/MinimumMinimum2quant_conv2d_2/MovingAvgQuantize/BatchMin:output:03quant_conv2d_2/MovingAvgQuantize/Minimum/y:output:0*
T0*
_output_shapes
: o
*quant_conv2d_2/MovingAvgQuantize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ½
(quant_conv2d_2/MovingAvgQuantize/MaximumMaximum2quant_conv2d_2/MovingAvgQuantize/BatchMax:output:03quant_conv2d_2/MovingAvgQuantize/Maximum/y:output:0*
T0*
_output_shapes
: x
3quant_conv2d_2/MovingAvgQuantize/AssignMinEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o:º
<quant_conv2d_2/MovingAvgQuantize/AssignMinEma/ReadVariableOpReadVariableOpEquant_conv2d_2_movingavgquantize_assignminema_readvariableop_resource*
_output_shapes
: *
dtype0Í
1quant_conv2d_2/MovingAvgQuantize/AssignMinEma/subSubDquant_conv2d_2/MovingAvgQuantize/AssignMinEma/ReadVariableOp:value:0,quant_conv2d_2/MovingAvgQuantize/Minimum:z:0*
T0*
_output_shapes
: Î
1quant_conv2d_2/MovingAvgQuantize/AssignMinEma/mulMul5quant_conv2d_2/MovingAvgQuantize/AssignMinEma/sub:z:0<quant_conv2d_2/MovingAvgQuantize/AssignMinEma/decay:output:0*
T0*
_output_shapes
: ¸
Aquant_conv2d_2/MovingAvgQuantize/AssignMinEma/AssignSubVariableOpAssignSubVariableOpEquant_conv2d_2_movingavgquantize_assignminema_readvariableop_resource5quant_conv2d_2/MovingAvgQuantize/AssignMinEma/mul:z:0=^quant_conv2d_2/MovingAvgQuantize/AssignMinEma/ReadVariableOp*
_output_shapes
 *
dtype0x
3quant_conv2d_2/MovingAvgQuantize/AssignMaxEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o:º
<quant_conv2d_2/MovingAvgQuantize/AssignMaxEma/ReadVariableOpReadVariableOpEquant_conv2d_2_movingavgquantize_assignmaxema_readvariableop_resource*
_output_shapes
: *
dtype0Í
1quant_conv2d_2/MovingAvgQuantize/AssignMaxEma/subSubDquant_conv2d_2/MovingAvgQuantize/AssignMaxEma/ReadVariableOp:value:0,quant_conv2d_2/MovingAvgQuantize/Maximum:z:0*
T0*
_output_shapes
: Î
1quant_conv2d_2/MovingAvgQuantize/AssignMaxEma/mulMul5quant_conv2d_2/MovingAvgQuantize/AssignMaxEma/sub:z:0<quant_conv2d_2/MovingAvgQuantize/AssignMaxEma/decay:output:0*
T0*
_output_shapes
: ¸
Aquant_conv2d_2/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpAssignSubVariableOpEquant_conv2d_2_movingavgquantize_assignmaxema_readvariableop_resource5quant_conv2d_2/MovingAvgQuantize/AssignMaxEma/mul:z:0=^quant_conv2d_2/MovingAvgQuantize/AssignMaxEma/ReadVariableOp*
_output_shapes
 *
dtype0
Gquant_conv2d_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpEquant_conv2d_2_movingavgquantize_assignminema_readvariableop_resourceB^quant_conv2d_2/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp*
_output_shapes
: *
dtype0
Iquant_conv2d_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpEquant_conv2d_2_movingavgquantize_assignmaxema_readvariableop_resourceB^quant_conv2d_2/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp*
_output_shapes
: *
dtype0Ë
8quant_conv2d_2/MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars!quant_conv2d_2/Relu:activations:0Oquant_conv2d_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Qquant_conv2d_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¼
5quant_conv2d_3/LastValueQuant/BatchMin/ReadVariableOpReadVariableOp>quant_conv2d_3_lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:@@*
dtype0
8quant_conv2d_3/LastValueQuant/BatchMin/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          Ô
&quant_conv2d_3/LastValueQuant/BatchMinMin=quant_conv2d_3/LastValueQuant/BatchMin/ReadVariableOp:value:0Aquant_conv2d_3/LastValueQuant/BatchMin/reduction_indices:output:0*
T0*
_output_shapes
:@¼
5quant_conv2d_3/LastValueQuant/BatchMax/ReadVariableOpReadVariableOp>quant_conv2d_3_lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:@@*
dtype0
8quant_conv2d_3/LastValueQuant/BatchMax/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          Ô
&quant_conv2d_3/LastValueQuant/BatchMaxMax=quant_conv2d_3/LastValueQuant/BatchMax/ReadVariableOp:value:0Aquant_conv2d_3/LastValueQuant/BatchMax/reduction_indices:output:0*
T0*
_output_shapes
:@l
'quant_conv2d_3/LastValueQuant/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ¿¸
%quant_conv2d_3/LastValueQuant/truedivRealDiv/quant_conv2d_3/LastValueQuant/BatchMax:output:00quant_conv2d_3/LastValueQuant/truediv/y:output:0*
T0*
_output_shapes
:@±
%quant_conv2d_3/LastValueQuant/MinimumMinimum/quant_conv2d_3/LastValueQuant/BatchMin:output:0)quant_conv2d_3/LastValueQuant/truediv:z:0*
T0*
_output_shapes
:@h
#quant_conv2d_3/LastValueQuant/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ¿¬
!quant_conv2d_3/LastValueQuant/mulMul/quant_conv2d_3/LastValueQuant/BatchMin:output:0,quant_conv2d_3/LastValueQuant/mul/y:output:0*
T0*
_output_shapes
:@­
%quant_conv2d_3/LastValueQuant/MaximumMaximum/quant_conv2d_3/LastValueQuant/BatchMax:output:0%quant_conv2d_3/LastValueQuant/mul:z:0*
T0*
_output_shapes
:@Ù
+quant_conv2d_3/LastValueQuant/AssignMinLastAssignVariableOp4quant_conv2d_3_lastvaluequant_assignminlast_resource)quant_conv2d_3/LastValueQuant/Minimum:z:0*
_output_shapes
 *
dtype0*
validate_shape(Ù
+quant_conv2d_3/LastValueQuant/AssignMaxLastAssignVariableOp4quant_conv2d_3_lastvaluequant_assignmaxlast_resource)quant_conv2d_3/LastValueQuant/Maximum:z:0*
_output_shapes
 *
dtype0*
validate_shape(Õ
Nquant_conv2d_3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpReadVariableOp>quant_conv2d_3_lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:@@*
dtype0ï
Pquant_conv2d_3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ReadVariableOp4quant_conv2d_3_lastvaluequant_assignminlast_resource,^quant_conv2d_3/LastValueQuant/AssignMinLast*
_output_shapes
:@*
dtype0ï
Pquant_conv2d_3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ReadVariableOp4quant_conv2d_3_lastvaluequant_assignmaxlast_resource,^quant_conv2d_3/LastValueQuant/AssignMaxLast*
_output_shapes
:@*
dtype0¬
?quant_conv2d_3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel!FakeQuantWithMinMaxVarsPerChannelVquant_conv2d_3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp:value:0Xquant_conv2d_3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1:value:0Xquant_conv2d_3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2:value:0*&
_output_shapes
:@@*
narrow_range(
quant_conv2d_3/Conv2DConv2DBquant_conv2d_2/MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0Iquant_conv2d_3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel:outputs:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingSAME*
strides

%quant_conv2d_3/BiasAdd/ReadVariableOpReadVariableOp.quant_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0ª
quant_conv2d_3/BiasAddBiasAddquant_conv2d_3/Conv2D:output:0-quant_conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@v
quant_conv2d_3/ReluReluquant_conv2d_3/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
&quant_conv2d_3/MovingAvgQuantize/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             ¥
)quant_conv2d_3/MovingAvgQuantize/BatchMinMin!quant_conv2d_3/Relu:activations:0/quant_conv2d_3/MovingAvgQuantize/Const:output:0*
T0*
_output_shapes
: 
(quant_conv2d_3/MovingAvgQuantize/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             §
)quant_conv2d_3/MovingAvgQuantize/BatchMaxMax!quant_conv2d_3/Relu:activations:01quant_conv2d_3/MovingAvgQuantize/Const_1:output:0*
T0*
_output_shapes
: o
*quant_conv2d_3/MovingAvgQuantize/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ½
(quant_conv2d_3/MovingAvgQuantize/MinimumMinimum2quant_conv2d_3/MovingAvgQuantize/BatchMin:output:03quant_conv2d_3/MovingAvgQuantize/Minimum/y:output:0*
T0*
_output_shapes
: o
*quant_conv2d_3/MovingAvgQuantize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ½
(quant_conv2d_3/MovingAvgQuantize/MaximumMaximum2quant_conv2d_3/MovingAvgQuantize/BatchMax:output:03quant_conv2d_3/MovingAvgQuantize/Maximum/y:output:0*
T0*
_output_shapes
: x
3quant_conv2d_3/MovingAvgQuantize/AssignMinEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o:º
<quant_conv2d_3/MovingAvgQuantize/AssignMinEma/ReadVariableOpReadVariableOpEquant_conv2d_3_movingavgquantize_assignminema_readvariableop_resource*
_output_shapes
: *
dtype0Í
1quant_conv2d_3/MovingAvgQuantize/AssignMinEma/subSubDquant_conv2d_3/MovingAvgQuantize/AssignMinEma/ReadVariableOp:value:0,quant_conv2d_3/MovingAvgQuantize/Minimum:z:0*
T0*
_output_shapes
: Î
1quant_conv2d_3/MovingAvgQuantize/AssignMinEma/mulMul5quant_conv2d_3/MovingAvgQuantize/AssignMinEma/sub:z:0<quant_conv2d_3/MovingAvgQuantize/AssignMinEma/decay:output:0*
T0*
_output_shapes
: ¸
Aquant_conv2d_3/MovingAvgQuantize/AssignMinEma/AssignSubVariableOpAssignSubVariableOpEquant_conv2d_3_movingavgquantize_assignminema_readvariableop_resource5quant_conv2d_3/MovingAvgQuantize/AssignMinEma/mul:z:0=^quant_conv2d_3/MovingAvgQuantize/AssignMinEma/ReadVariableOp*
_output_shapes
 *
dtype0x
3quant_conv2d_3/MovingAvgQuantize/AssignMaxEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o:º
<quant_conv2d_3/MovingAvgQuantize/AssignMaxEma/ReadVariableOpReadVariableOpEquant_conv2d_3_movingavgquantize_assignmaxema_readvariableop_resource*
_output_shapes
: *
dtype0Í
1quant_conv2d_3/MovingAvgQuantize/AssignMaxEma/subSubDquant_conv2d_3/MovingAvgQuantize/AssignMaxEma/ReadVariableOp:value:0,quant_conv2d_3/MovingAvgQuantize/Maximum:z:0*
T0*
_output_shapes
: Î
1quant_conv2d_3/MovingAvgQuantize/AssignMaxEma/mulMul5quant_conv2d_3/MovingAvgQuantize/AssignMaxEma/sub:z:0<quant_conv2d_3/MovingAvgQuantize/AssignMaxEma/decay:output:0*
T0*
_output_shapes
: ¸
Aquant_conv2d_3/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpAssignSubVariableOpEquant_conv2d_3_movingavgquantize_assignmaxema_readvariableop_resource5quant_conv2d_3/MovingAvgQuantize/AssignMaxEma/mul:z:0=^quant_conv2d_3/MovingAvgQuantize/AssignMaxEma/ReadVariableOp*
_output_shapes
 *
dtype0
Gquant_conv2d_3/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpEquant_conv2d_3_movingavgquantize_assignminema_readvariableop_resourceB^quant_conv2d_3/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp*
_output_shapes
: *
dtype0
Iquant_conv2d_3/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpEquant_conv2d_3_movingavgquantize_assignmaxema_readvariableop_resourceB^quant_conv2d_3/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp*
_output_shapes
: *
dtype0Ë
8quant_conv2d_3/MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars!quant_conv2d_3/Relu:activations:0Oquant_conv2d_3/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Qquant_conv2d_3/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¼
5quant_conv2d_4/LastValueQuant/BatchMin/ReadVariableOpReadVariableOp>quant_conv2d_4_lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:@ *
dtype0
8quant_conv2d_4/LastValueQuant/BatchMin/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          Ô
&quant_conv2d_4/LastValueQuant/BatchMinMin=quant_conv2d_4/LastValueQuant/BatchMin/ReadVariableOp:value:0Aquant_conv2d_4/LastValueQuant/BatchMin/reduction_indices:output:0*
T0*
_output_shapes
: ¼
5quant_conv2d_4/LastValueQuant/BatchMax/ReadVariableOpReadVariableOp>quant_conv2d_4_lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:@ *
dtype0
8quant_conv2d_4/LastValueQuant/BatchMax/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          Ô
&quant_conv2d_4/LastValueQuant/BatchMaxMax=quant_conv2d_4/LastValueQuant/BatchMax/ReadVariableOp:value:0Aquant_conv2d_4/LastValueQuant/BatchMax/reduction_indices:output:0*
T0*
_output_shapes
: l
'quant_conv2d_4/LastValueQuant/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ¿¸
%quant_conv2d_4/LastValueQuant/truedivRealDiv/quant_conv2d_4/LastValueQuant/BatchMax:output:00quant_conv2d_4/LastValueQuant/truediv/y:output:0*
T0*
_output_shapes
: ±
%quant_conv2d_4/LastValueQuant/MinimumMinimum/quant_conv2d_4/LastValueQuant/BatchMin:output:0)quant_conv2d_4/LastValueQuant/truediv:z:0*
T0*
_output_shapes
: h
#quant_conv2d_4/LastValueQuant/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ¿¬
!quant_conv2d_4/LastValueQuant/mulMul/quant_conv2d_4/LastValueQuant/BatchMin:output:0,quant_conv2d_4/LastValueQuant/mul/y:output:0*
T0*
_output_shapes
: ­
%quant_conv2d_4/LastValueQuant/MaximumMaximum/quant_conv2d_4/LastValueQuant/BatchMax:output:0%quant_conv2d_4/LastValueQuant/mul:z:0*
T0*
_output_shapes
: Ù
+quant_conv2d_4/LastValueQuant/AssignMinLastAssignVariableOp4quant_conv2d_4_lastvaluequant_assignminlast_resource)quant_conv2d_4/LastValueQuant/Minimum:z:0*
_output_shapes
 *
dtype0*
validate_shape(Ù
+quant_conv2d_4/LastValueQuant/AssignMaxLastAssignVariableOp4quant_conv2d_4_lastvaluequant_assignmaxlast_resource)quant_conv2d_4/LastValueQuant/Maximum:z:0*
_output_shapes
 *
dtype0*
validate_shape(Õ
Nquant_conv2d_4/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpReadVariableOp>quant_conv2d_4_lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:@ *
dtype0ï
Pquant_conv2d_4/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ReadVariableOp4quant_conv2d_4_lastvaluequant_assignminlast_resource,^quant_conv2d_4/LastValueQuant/AssignMinLast*
_output_shapes
: *
dtype0ï
Pquant_conv2d_4/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ReadVariableOp4quant_conv2d_4_lastvaluequant_assignmaxlast_resource,^quant_conv2d_4/LastValueQuant/AssignMaxLast*
_output_shapes
: *
dtype0¬
?quant_conv2d_4/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel!FakeQuantWithMinMaxVarsPerChannelVquant_conv2d_4/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp:value:0Xquant_conv2d_4/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1:value:0Xquant_conv2d_4/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2:value:0*&
_output_shapes
:@ *
narrow_range(
quant_conv2d_4/Conv2DConv2DBquant_conv2d_3/MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0Iquant_conv2d_4/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel:outputs:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides

%quant_conv2d_4/BiasAdd/ReadVariableOpReadVariableOp.quant_conv2d_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0ª
quant_conv2d_4/BiasAddBiasAddquant_conv2d_4/Conv2D:output:0-quant_conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ v
quant_conv2d_4/ReluReluquant_conv2d_4/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
&quant_conv2d_4/MovingAvgQuantize/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             ¥
)quant_conv2d_4/MovingAvgQuantize/BatchMinMin!quant_conv2d_4/Relu:activations:0/quant_conv2d_4/MovingAvgQuantize/Const:output:0*
T0*
_output_shapes
: 
(quant_conv2d_4/MovingAvgQuantize/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             §
)quant_conv2d_4/MovingAvgQuantize/BatchMaxMax!quant_conv2d_4/Relu:activations:01quant_conv2d_4/MovingAvgQuantize/Const_1:output:0*
T0*
_output_shapes
: o
*quant_conv2d_4/MovingAvgQuantize/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ½
(quant_conv2d_4/MovingAvgQuantize/MinimumMinimum2quant_conv2d_4/MovingAvgQuantize/BatchMin:output:03quant_conv2d_4/MovingAvgQuantize/Minimum/y:output:0*
T0*
_output_shapes
: o
*quant_conv2d_4/MovingAvgQuantize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ½
(quant_conv2d_4/MovingAvgQuantize/MaximumMaximum2quant_conv2d_4/MovingAvgQuantize/BatchMax:output:03quant_conv2d_4/MovingAvgQuantize/Maximum/y:output:0*
T0*
_output_shapes
: x
3quant_conv2d_4/MovingAvgQuantize/AssignMinEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o:º
<quant_conv2d_4/MovingAvgQuantize/AssignMinEma/ReadVariableOpReadVariableOpEquant_conv2d_4_movingavgquantize_assignminema_readvariableop_resource*
_output_shapes
: *
dtype0Í
1quant_conv2d_4/MovingAvgQuantize/AssignMinEma/subSubDquant_conv2d_4/MovingAvgQuantize/AssignMinEma/ReadVariableOp:value:0,quant_conv2d_4/MovingAvgQuantize/Minimum:z:0*
T0*
_output_shapes
: Î
1quant_conv2d_4/MovingAvgQuantize/AssignMinEma/mulMul5quant_conv2d_4/MovingAvgQuantize/AssignMinEma/sub:z:0<quant_conv2d_4/MovingAvgQuantize/AssignMinEma/decay:output:0*
T0*
_output_shapes
: ¸
Aquant_conv2d_4/MovingAvgQuantize/AssignMinEma/AssignSubVariableOpAssignSubVariableOpEquant_conv2d_4_movingavgquantize_assignminema_readvariableop_resource5quant_conv2d_4/MovingAvgQuantize/AssignMinEma/mul:z:0=^quant_conv2d_4/MovingAvgQuantize/AssignMinEma/ReadVariableOp*
_output_shapes
 *
dtype0x
3quant_conv2d_4/MovingAvgQuantize/AssignMaxEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o:º
<quant_conv2d_4/MovingAvgQuantize/AssignMaxEma/ReadVariableOpReadVariableOpEquant_conv2d_4_movingavgquantize_assignmaxema_readvariableop_resource*
_output_shapes
: *
dtype0Í
1quant_conv2d_4/MovingAvgQuantize/AssignMaxEma/subSubDquant_conv2d_4/MovingAvgQuantize/AssignMaxEma/ReadVariableOp:value:0,quant_conv2d_4/MovingAvgQuantize/Maximum:z:0*
T0*
_output_shapes
: Î
1quant_conv2d_4/MovingAvgQuantize/AssignMaxEma/mulMul5quant_conv2d_4/MovingAvgQuantize/AssignMaxEma/sub:z:0<quant_conv2d_4/MovingAvgQuantize/AssignMaxEma/decay:output:0*
T0*
_output_shapes
: ¸
Aquant_conv2d_4/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpAssignSubVariableOpEquant_conv2d_4_movingavgquantize_assignmaxema_readvariableop_resource5quant_conv2d_4/MovingAvgQuantize/AssignMaxEma/mul:z:0=^quant_conv2d_4/MovingAvgQuantize/AssignMaxEma/ReadVariableOp*
_output_shapes
 *
dtype0
Gquant_conv2d_4/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpEquant_conv2d_4_movingavgquantize_assignminema_readvariableop_resourceB^quant_conv2d_4/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp*
_output_shapes
: *
dtype0
Iquant_conv2d_4/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpEquant_conv2d_4_movingavgquantize_assignmaxema_readvariableop_resourceB^quant_conv2d_4/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp*
_output_shapes
: *
dtype0Ë
8quant_conv2d_4/MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars!quant_conv2d_4/Relu:activations:0Oquant_conv2d_4/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Qquant_conv2d_4/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Ù
quant_max_pooling2d_2/MaxPoolMaxPoolBquant_conv2d_4/MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
ksize
*
paddingVALID*
strides
d
quant_flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   
quant_flatten/ReshapeReshape&quant_max_pooling2d_2/MaxPool:output:0quant_flatten/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿq
 quant_dense/LastValueQuant/ConstConst*
_output_shapes
:*
dtype0*
valueB"       °
2quant_dense/LastValueQuant/BatchMin/ReadVariableOpReadVariableOp;quant_dense_lastvaluequant_batchmin_readvariableop_resource* 
_output_shapes
:
*
dtype0²
#quant_dense/LastValueQuant/BatchMinMin:quant_dense/LastValueQuant/BatchMin/ReadVariableOp:value:0)quant_dense/LastValueQuant/Const:output:0*
T0*
_output_shapes
: s
"quant_dense/LastValueQuant/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       °
2quant_dense/LastValueQuant/BatchMax/ReadVariableOpReadVariableOp;quant_dense_lastvaluequant_batchmin_readvariableop_resource* 
_output_shapes
:
*
dtype0´
#quant_dense/LastValueQuant/BatchMaxMax:quant_dense/LastValueQuant/BatchMax/ReadVariableOp:value:0+quant_dense/LastValueQuant/Const_1:output:0*
T0*
_output_shapes
: i
$quant_dense/LastValueQuant/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ¿«
"quant_dense/LastValueQuant/truedivRealDiv,quant_dense/LastValueQuant/BatchMax:output:0-quant_dense/LastValueQuant/truediv/y:output:0*
T0*
_output_shapes
: ¤
"quant_dense/LastValueQuant/MinimumMinimum,quant_dense/LastValueQuant/BatchMin:output:0&quant_dense/LastValueQuant/truediv:z:0*
T0*
_output_shapes
: e
 quant_dense/LastValueQuant/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ¿
quant_dense/LastValueQuant/mulMul,quant_dense/LastValueQuant/BatchMin:output:0)quant_dense/LastValueQuant/mul/y:output:0*
T0*
_output_shapes
:  
"quant_dense/LastValueQuant/MaximumMaximum,quant_dense/LastValueQuant/BatchMax:output:0"quant_dense/LastValueQuant/mul:z:0*
T0*
_output_shapes
: Ð
(quant_dense/LastValueQuant/AssignMinLastAssignVariableOp1quant_dense_lastvaluequant_assignminlast_resource&quant_dense/LastValueQuant/Minimum:z:0*
_output_shapes
 *
dtype0*
validate_shape(Ð
(quant_dense/LastValueQuant/AssignMaxLastAssignVariableOp1quant_dense_lastvaluequant_assignmaxlast_resource&quant_dense/LastValueQuant/Maximum:z:0*
_output_shapes
 *
dtype0*
validate_shape(¿
Aquant_dense/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp;quant_dense_lastvaluequant_batchmin_readvariableop_resource* 
_output_shapes
:
*
dtype0Ø
Cquant_dense/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp1quant_dense_lastvaluequant_assignminlast_resource)^quant_dense/LastValueQuant/AssignMinLast*
_output_shapes
: *
dtype0Ø
Cquant_dense/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2ReadVariableOp1quant_dense_lastvaluequant_assignmaxlast_resource)^quant_dense/LastValueQuant/AssignMaxLast*
_output_shapes
: *
dtype0è
2quant_dense/LastValueQuant/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsIquant_dense/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Kquant_dense/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0Kquant_dense/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2:value:0* 
_output_shapes
:
*
narrow_range(­
quant_dense/MatMulMatMulquant_flatten/Reshape:output:0<quant_dense/LastValueQuant/FakeQuantWithMinMaxVars:outputs:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"quant_dense/BiasAdd/ReadVariableOpReadVariableOp+quant_dense_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
quant_dense/BiasAddBiasAddquant_dense/MatMul:product:0*quant_dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
quant_dense/ReluReluquant_dense/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿt
#quant_dense/MovingAvgQuantize/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
&quant_dense/MovingAvgQuantize/BatchMinMinquant_dense/Relu:activations:0,quant_dense/MovingAvgQuantize/Const:output:0*
T0*
_output_shapes
: v
%quant_dense/MovingAvgQuantize/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       
&quant_dense/MovingAvgQuantize/BatchMaxMaxquant_dense/Relu:activations:0.quant_dense/MovingAvgQuantize/Const_1:output:0*
T0*
_output_shapes
: l
'quant_dense/MovingAvgQuantize/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ´
%quant_dense/MovingAvgQuantize/MinimumMinimum/quant_dense/MovingAvgQuantize/BatchMin:output:00quant_dense/MovingAvgQuantize/Minimum/y:output:0*
T0*
_output_shapes
: l
'quant_dense/MovingAvgQuantize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ´
%quant_dense/MovingAvgQuantize/MaximumMaximum/quant_dense/MovingAvgQuantize/BatchMax:output:00quant_dense/MovingAvgQuantize/Maximum/y:output:0*
T0*
_output_shapes
: u
0quant_dense/MovingAvgQuantize/AssignMinEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o:´
9quant_dense/MovingAvgQuantize/AssignMinEma/ReadVariableOpReadVariableOpBquant_dense_movingavgquantize_assignminema_readvariableop_resource*
_output_shapes
: *
dtype0Ä
.quant_dense/MovingAvgQuantize/AssignMinEma/subSubAquant_dense/MovingAvgQuantize/AssignMinEma/ReadVariableOp:value:0)quant_dense/MovingAvgQuantize/Minimum:z:0*
T0*
_output_shapes
: Å
.quant_dense/MovingAvgQuantize/AssignMinEma/mulMul2quant_dense/MovingAvgQuantize/AssignMinEma/sub:z:09quant_dense/MovingAvgQuantize/AssignMinEma/decay:output:0*
T0*
_output_shapes
: ¬
>quant_dense/MovingAvgQuantize/AssignMinEma/AssignSubVariableOpAssignSubVariableOpBquant_dense_movingavgquantize_assignminema_readvariableop_resource2quant_dense/MovingAvgQuantize/AssignMinEma/mul:z:0:^quant_dense/MovingAvgQuantize/AssignMinEma/ReadVariableOp*
_output_shapes
 *
dtype0u
0quant_dense/MovingAvgQuantize/AssignMaxEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o:´
9quant_dense/MovingAvgQuantize/AssignMaxEma/ReadVariableOpReadVariableOpBquant_dense_movingavgquantize_assignmaxema_readvariableop_resource*
_output_shapes
: *
dtype0Ä
.quant_dense/MovingAvgQuantize/AssignMaxEma/subSubAquant_dense/MovingAvgQuantize/AssignMaxEma/ReadVariableOp:value:0)quant_dense/MovingAvgQuantize/Maximum:z:0*
T0*
_output_shapes
: Å
.quant_dense/MovingAvgQuantize/AssignMaxEma/mulMul2quant_dense/MovingAvgQuantize/AssignMaxEma/sub:z:09quant_dense/MovingAvgQuantize/AssignMaxEma/decay:output:0*
T0*
_output_shapes
: ¬
>quant_dense/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpAssignSubVariableOpBquant_dense_movingavgquantize_assignmaxema_readvariableop_resource2quant_dense/MovingAvgQuantize/AssignMaxEma/mul:z:0:^quant_dense/MovingAvgQuantize/AssignMaxEma/ReadVariableOp*
_output_shapes
 *
dtype0
Dquant_dense/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpBquant_dense_movingavgquantize_assignminema_readvariableop_resource?^quant_dense/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp*
_output_shapes
: *
dtype0
Fquant_dense/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpBquant_dense_movingavgquantize_assignmaxema_readvariableop_resource?^quant_dense/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp*
_output_shapes
: *
dtype0¸
5quant_dense/MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsquant_dense/Relu:activations:0Lquant_dense/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Nquant_dense/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
quant_dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @º
quant_dropout/dropout/MulMul?quant_dense/MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0$quant_dropout/dropout/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
quant_dropout/dropout/ShapeShape?quant_dense/MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0*
T0*
_output_shapes
:©
2quant_dropout/dropout/random_uniform/RandomUniformRandomUniform$quant_dropout/dropout/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0i
$quant_dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Ñ
"quant_dropout/dropout/GreaterEqualGreaterEqual;quant_dropout/dropout/random_uniform/RandomUniform:output:0-quant_dropout/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
quant_dropout/dropout/CastCast&quant_dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
quant_dropout/dropout/Mul_1Mulquant_dropout/dropout/Mul:z:0quant_dropout/dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
"quant_dense_1/LastValueQuant/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ´
4quant_dense_1/LastValueQuant/BatchMin/ReadVariableOpReadVariableOp=quant_dense_1_lastvaluequant_batchmin_readvariableop_resource* 
_output_shapes
:
*
dtype0¸
%quant_dense_1/LastValueQuant/BatchMinMin<quant_dense_1/LastValueQuant/BatchMin/ReadVariableOp:value:0+quant_dense_1/LastValueQuant/Const:output:0*
T0*
_output_shapes
: u
$quant_dense_1/LastValueQuant/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ´
4quant_dense_1/LastValueQuant/BatchMax/ReadVariableOpReadVariableOp=quant_dense_1_lastvaluequant_batchmin_readvariableop_resource* 
_output_shapes
:
*
dtype0º
%quant_dense_1/LastValueQuant/BatchMaxMax<quant_dense_1/LastValueQuant/BatchMax/ReadVariableOp:value:0-quant_dense_1/LastValueQuant/Const_1:output:0*
T0*
_output_shapes
: k
&quant_dense_1/LastValueQuant/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ¿±
$quant_dense_1/LastValueQuant/truedivRealDiv.quant_dense_1/LastValueQuant/BatchMax:output:0/quant_dense_1/LastValueQuant/truediv/y:output:0*
T0*
_output_shapes
: ª
$quant_dense_1/LastValueQuant/MinimumMinimum.quant_dense_1/LastValueQuant/BatchMin:output:0(quant_dense_1/LastValueQuant/truediv:z:0*
T0*
_output_shapes
: g
"quant_dense_1/LastValueQuant/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ¿¥
 quant_dense_1/LastValueQuant/mulMul.quant_dense_1/LastValueQuant/BatchMin:output:0+quant_dense_1/LastValueQuant/mul/y:output:0*
T0*
_output_shapes
: ¦
$quant_dense_1/LastValueQuant/MaximumMaximum.quant_dense_1/LastValueQuant/BatchMax:output:0$quant_dense_1/LastValueQuant/mul:z:0*
T0*
_output_shapes
: Ö
*quant_dense_1/LastValueQuant/AssignMinLastAssignVariableOp3quant_dense_1_lastvaluequant_assignminlast_resource(quant_dense_1/LastValueQuant/Minimum:z:0*
_output_shapes
 *
dtype0*
validate_shape(Ö
*quant_dense_1/LastValueQuant/AssignMaxLastAssignVariableOp3quant_dense_1_lastvaluequant_assignmaxlast_resource(quant_dense_1/LastValueQuant/Maximum:z:0*
_output_shapes
 *
dtype0*
validate_shape(Ã
Cquant_dense_1/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp=quant_dense_1_lastvaluequant_batchmin_readvariableop_resource* 
_output_shapes
:
*
dtype0Þ
Equant_dense_1/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp3quant_dense_1_lastvaluequant_assignminlast_resource+^quant_dense_1/LastValueQuant/AssignMinLast*
_output_shapes
: *
dtype0Þ
Equant_dense_1/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2ReadVariableOp3quant_dense_1_lastvaluequant_assignmaxlast_resource+^quant_dense_1/LastValueQuant/AssignMaxLast*
_output_shapes
: *
dtype0ð
4quant_dense_1/LastValueQuant/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsKquant_dense_1/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Mquant_dense_1/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0Mquant_dense_1/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2:value:0* 
_output_shapes
:
*
narrow_range(²
quant_dense_1/MatMulMatMulquant_dropout/dropout/Mul_1:z:0>quant_dense_1/LastValueQuant/FakeQuantWithMinMaxVars:outputs:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$quant_dense_1/BiasAdd/ReadVariableOpReadVariableOp-quant_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0¡
quant_dense_1/BiasAddBiasAddquant_dense_1/MatMul:product:0,quant_dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿm
quant_dense_1/ReluReluquant_dense_1/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿv
%quant_dense_1/MovingAvgQuantize/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ¢
(quant_dense_1/MovingAvgQuantize/BatchMinMin quant_dense_1/Relu:activations:0.quant_dense_1/MovingAvgQuantize/Const:output:0*
T0*
_output_shapes
: x
'quant_dense_1/MovingAvgQuantize/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ¤
(quant_dense_1/MovingAvgQuantize/BatchMaxMax quant_dense_1/Relu:activations:00quant_dense_1/MovingAvgQuantize/Const_1:output:0*
T0*
_output_shapes
: n
)quant_dense_1/MovingAvgQuantize/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    º
'quant_dense_1/MovingAvgQuantize/MinimumMinimum1quant_dense_1/MovingAvgQuantize/BatchMin:output:02quant_dense_1/MovingAvgQuantize/Minimum/y:output:0*
T0*
_output_shapes
: n
)quant_dense_1/MovingAvgQuantize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    º
'quant_dense_1/MovingAvgQuantize/MaximumMaximum1quant_dense_1/MovingAvgQuantize/BatchMax:output:02quant_dense_1/MovingAvgQuantize/Maximum/y:output:0*
T0*
_output_shapes
: w
2quant_dense_1/MovingAvgQuantize/AssignMinEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o:¸
;quant_dense_1/MovingAvgQuantize/AssignMinEma/ReadVariableOpReadVariableOpDquant_dense_1_movingavgquantize_assignminema_readvariableop_resource*
_output_shapes
: *
dtype0Ê
0quant_dense_1/MovingAvgQuantize/AssignMinEma/subSubCquant_dense_1/MovingAvgQuantize/AssignMinEma/ReadVariableOp:value:0+quant_dense_1/MovingAvgQuantize/Minimum:z:0*
T0*
_output_shapes
: Ë
0quant_dense_1/MovingAvgQuantize/AssignMinEma/mulMul4quant_dense_1/MovingAvgQuantize/AssignMinEma/sub:z:0;quant_dense_1/MovingAvgQuantize/AssignMinEma/decay:output:0*
T0*
_output_shapes
: ´
@quant_dense_1/MovingAvgQuantize/AssignMinEma/AssignSubVariableOpAssignSubVariableOpDquant_dense_1_movingavgquantize_assignminema_readvariableop_resource4quant_dense_1/MovingAvgQuantize/AssignMinEma/mul:z:0<^quant_dense_1/MovingAvgQuantize/AssignMinEma/ReadVariableOp*
_output_shapes
 *
dtype0w
2quant_dense_1/MovingAvgQuantize/AssignMaxEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o:¸
;quant_dense_1/MovingAvgQuantize/AssignMaxEma/ReadVariableOpReadVariableOpDquant_dense_1_movingavgquantize_assignmaxema_readvariableop_resource*
_output_shapes
: *
dtype0Ê
0quant_dense_1/MovingAvgQuantize/AssignMaxEma/subSubCquant_dense_1/MovingAvgQuantize/AssignMaxEma/ReadVariableOp:value:0+quant_dense_1/MovingAvgQuantize/Maximum:z:0*
T0*
_output_shapes
: Ë
0quant_dense_1/MovingAvgQuantize/AssignMaxEma/mulMul4quant_dense_1/MovingAvgQuantize/AssignMaxEma/sub:z:0;quant_dense_1/MovingAvgQuantize/AssignMaxEma/decay:output:0*
T0*
_output_shapes
: ´
@quant_dense_1/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpAssignSubVariableOpDquant_dense_1_movingavgquantize_assignmaxema_readvariableop_resource4quant_dense_1/MovingAvgQuantize/AssignMaxEma/mul:z:0<^quant_dense_1/MovingAvgQuantize/AssignMaxEma/ReadVariableOp*
_output_shapes
 *
dtype0
Fquant_dense_1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpDquant_dense_1_movingavgquantize_assignminema_readvariableop_resourceA^quant_dense_1/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp*
_output_shapes
: *
dtype0
Hquant_dense_1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpDquant_dense_1_movingavgquantize_assignmaxema_readvariableop_resourceA^quant_dense_1/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp*
_output_shapes
: *
dtype0À
7quant_dense_1/MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars quant_dense_1/Relu:activations:0Nquant_dense_1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Pquant_dense_1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
quant_dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @À
quant_dropout_1/dropout/MulMulAquant_dense_1/MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0&quant_dropout_1/dropout/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
quant_dropout_1/dropout/ShapeShapeAquant_dense_1/MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0*
T0*
_output_shapes
:­
4quant_dropout_1/dropout/random_uniform/RandomUniformRandomUniform&quant_dropout_1/dropout/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0k
&quant_dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?×
$quant_dropout_1/dropout/GreaterEqualGreaterEqual=quant_dropout_1/dropout/random_uniform/RandomUniform:output:0/quant_dropout_1/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
quant_dropout_1/dropout/CastCast(quant_dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
quant_dropout_1/dropout/Mul_1Mulquant_dropout_1/dropout/Mul:z:0 quant_dropout_1/dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
"quant_dense_2/LastValueQuant/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ³
4quant_dense_2/LastValueQuant/BatchMin/ReadVariableOpReadVariableOp=quant_dense_2_lastvaluequant_batchmin_readvariableop_resource*
_output_shapes
:	*
dtype0¸
%quant_dense_2/LastValueQuant/BatchMinMin<quant_dense_2/LastValueQuant/BatchMin/ReadVariableOp:value:0+quant_dense_2/LastValueQuant/Const:output:0*
T0*
_output_shapes
: u
$quant_dense_2/LastValueQuant/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ³
4quant_dense_2/LastValueQuant/BatchMax/ReadVariableOpReadVariableOp=quant_dense_2_lastvaluequant_batchmin_readvariableop_resource*
_output_shapes
:	*
dtype0º
%quant_dense_2/LastValueQuant/BatchMaxMax<quant_dense_2/LastValueQuant/BatchMax/ReadVariableOp:value:0-quant_dense_2/LastValueQuant/Const_1:output:0*
T0*
_output_shapes
: k
&quant_dense_2/LastValueQuant/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ¿±
$quant_dense_2/LastValueQuant/truedivRealDiv.quant_dense_2/LastValueQuant/BatchMax:output:0/quant_dense_2/LastValueQuant/truediv/y:output:0*
T0*
_output_shapes
: ª
$quant_dense_2/LastValueQuant/MinimumMinimum.quant_dense_2/LastValueQuant/BatchMin:output:0(quant_dense_2/LastValueQuant/truediv:z:0*
T0*
_output_shapes
: g
"quant_dense_2/LastValueQuant/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ¿¥
 quant_dense_2/LastValueQuant/mulMul.quant_dense_2/LastValueQuant/BatchMin:output:0+quant_dense_2/LastValueQuant/mul/y:output:0*
T0*
_output_shapes
: ¦
$quant_dense_2/LastValueQuant/MaximumMaximum.quant_dense_2/LastValueQuant/BatchMax:output:0$quant_dense_2/LastValueQuant/mul:z:0*
T0*
_output_shapes
: Ö
*quant_dense_2/LastValueQuant/AssignMinLastAssignVariableOp3quant_dense_2_lastvaluequant_assignminlast_resource(quant_dense_2/LastValueQuant/Minimum:z:0*
_output_shapes
 *
dtype0*
validate_shape(Ö
*quant_dense_2/LastValueQuant/AssignMaxLastAssignVariableOp3quant_dense_2_lastvaluequant_assignmaxlast_resource(quant_dense_2/LastValueQuant/Maximum:z:0*
_output_shapes
 *
dtype0*
validate_shape(Â
Cquant_dense_2/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp=quant_dense_2_lastvaluequant_batchmin_readvariableop_resource*
_output_shapes
:	*
dtype0Þ
Equant_dense_2/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp3quant_dense_2_lastvaluequant_assignminlast_resource+^quant_dense_2/LastValueQuant/AssignMinLast*
_output_shapes
: *
dtype0Þ
Equant_dense_2/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2ReadVariableOp3quant_dense_2_lastvaluequant_assignmaxlast_resource+^quant_dense_2/LastValueQuant/AssignMaxLast*
_output_shapes
: *
dtype0ï
4quant_dense_2/LastValueQuant/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsKquant_dense_2/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Mquant_dense_2/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0Mquant_dense_2/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2:value:0*
_output_shapes
:	*
narrow_range(³
quant_dense_2/MatMulMatMul!quant_dropout_1/dropout/Mul_1:z:0>quant_dense_2/LastValueQuant/FakeQuantWithMinMaxVars:outputs:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$quant_dense_2/BiasAdd/ReadVariableOpReadVariableOp-quant_dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0 
quant_dense_2/BiasAddBiasAddquant_dense_2/MatMul:product:0,quant_dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿv
%quant_dense_2/MovingAvgQuantize/ConstConst*
_output_shapes
:*
dtype0*
valueB"        
(quant_dense_2/MovingAvgQuantize/BatchMinMinquant_dense_2/BiasAdd:output:0.quant_dense_2/MovingAvgQuantize/Const:output:0*
T0*
_output_shapes
: x
'quant_dense_2/MovingAvgQuantize/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ¢
(quant_dense_2/MovingAvgQuantize/BatchMaxMaxquant_dense_2/BiasAdd:output:00quant_dense_2/MovingAvgQuantize/Const_1:output:0*
T0*
_output_shapes
: n
)quant_dense_2/MovingAvgQuantize/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    º
'quant_dense_2/MovingAvgQuantize/MinimumMinimum1quant_dense_2/MovingAvgQuantize/BatchMin:output:02quant_dense_2/MovingAvgQuantize/Minimum/y:output:0*
T0*
_output_shapes
: n
)quant_dense_2/MovingAvgQuantize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    º
'quant_dense_2/MovingAvgQuantize/MaximumMaximum1quant_dense_2/MovingAvgQuantize/BatchMax:output:02quant_dense_2/MovingAvgQuantize/Maximum/y:output:0*
T0*
_output_shapes
: w
2quant_dense_2/MovingAvgQuantize/AssignMinEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o:¸
;quant_dense_2/MovingAvgQuantize/AssignMinEma/ReadVariableOpReadVariableOpDquant_dense_2_movingavgquantize_assignminema_readvariableop_resource*
_output_shapes
: *
dtype0Ê
0quant_dense_2/MovingAvgQuantize/AssignMinEma/subSubCquant_dense_2/MovingAvgQuantize/AssignMinEma/ReadVariableOp:value:0+quant_dense_2/MovingAvgQuantize/Minimum:z:0*
T0*
_output_shapes
: Ë
0quant_dense_2/MovingAvgQuantize/AssignMinEma/mulMul4quant_dense_2/MovingAvgQuantize/AssignMinEma/sub:z:0;quant_dense_2/MovingAvgQuantize/AssignMinEma/decay:output:0*
T0*
_output_shapes
: ´
@quant_dense_2/MovingAvgQuantize/AssignMinEma/AssignSubVariableOpAssignSubVariableOpDquant_dense_2_movingavgquantize_assignminema_readvariableop_resource4quant_dense_2/MovingAvgQuantize/AssignMinEma/mul:z:0<^quant_dense_2/MovingAvgQuantize/AssignMinEma/ReadVariableOp*
_output_shapes
 *
dtype0w
2quant_dense_2/MovingAvgQuantize/AssignMaxEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o:¸
;quant_dense_2/MovingAvgQuantize/AssignMaxEma/ReadVariableOpReadVariableOpDquant_dense_2_movingavgquantize_assignmaxema_readvariableop_resource*
_output_shapes
: *
dtype0Ê
0quant_dense_2/MovingAvgQuantize/AssignMaxEma/subSubCquant_dense_2/MovingAvgQuantize/AssignMaxEma/ReadVariableOp:value:0+quant_dense_2/MovingAvgQuantize/Maximum:z:0*
T0*
_output_shapes
: Ë
0quant_dense_2/MovingAvgQuantize/AssignMaxEma/mulMul4quant_dense_2/MovingAvgQuantize/AssignMaxEma/sub:z:0;quant_dense_2/MovingAvgQuantize/AssignMaxEma/decay:output:0*
T0*
_output_shapes
: ´
@quant_dense_2/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpAssignSubVariableOpDquant_dense_2_movingavgquantize_assignmaxema_readvariableop_resource4quant_dense_2/MovingAvgQuantize/AssignMaxEma/mul:z:0<^quant_dense_2/MovingAvgQuantize/AssignMaxEma/ReadVariableOp*
_output_shapes
 *
dtype0
Fquant_dense_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpDquant_dense_2_movingavgquantize_assignminema_readvariableop_resourceA^quant_dense_2/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp*
_output_shapes
: *
dtype0
Hquant_dense_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpDquant_dense_2_movingavgquantize_assignmaxema_readvariableop_resourceA^quant_dense_2/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp*
_output_shapes
: *
dtype0½
7quant_dense_2/MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsquant_dense_2/BiasAdd:output:0Nquant_dense_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Pquant_dense_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
quant_dense_2/SoftmaxSoftmaxAquant_dense_2/MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´
/conv2d/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp<quant_conv2d_lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
: *
dtype0
 conv2d/kernel/Regularizer/L2LossL2Loss7conv2d/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: d
conv2d/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;
conv2d/kernel/Regularizer/mulMul(conv2d/kernel/Regularizer/mul/x:output:0)conv2d/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: ¸
1conv2d_1/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp>quant_conv2d_1_lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:  *
dtype0
"conv2d_1/kernel/Regularizer/L2LossL2Loss9conv2d_1/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!conv2d_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o; 
conv2d_1/kernel/Regularizer/mulMul*conv2d_1/kernel/Regularizer/mul/x:output:0+conv2d_1/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: ¸
1conv2d_2/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp>quant_conv2d_2_lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
: @*
dtype0
"conv2d_2/kernel/Regularizer/L2LossL2Loss9conv2d_2/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!conv2d_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o; 
conv2d_2/kernel/Regularizer/mulMul*conv2d_2/kernel/Regularizer/mul/x:output:0+conv2d_2/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: ¸
1conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp>quant_conv2d_3_lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:@@*
dtype0
"conv2d_3/kernel/Regularizer/L2LossL2Loss9conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!conv2d_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o; 
conv2d_3/kernel/Regularizer/mulMul*conv2d_3/kernel/Regularizer/mul/x:output:0+conv2d_3/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: ¸
1conv2d_4/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp>quant_conv2d_4_lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:@ *
dtype0
"conv2d_4/kernel/Regularizer/L2LossL2Loss9conv2d_4/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!conv2d_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o; 
conv2d_4/kernel/Regularizer/mulMul*conv2d_4/kernel/Regularizer/mul/x:output:0+conv2d_4/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: ¬
.dense/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp;quant_dense_lastvaluequant_batchmin_readvariableop_resource* 
_output_shapes
:
*
dtype0
dense/kernel/Regularizer/L2LossL2Loss6dense/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: c
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0(dense/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: °
0dense_1/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp=quant_dense_1_lastvaluequant_batchmin_readvariableop_resource* 
_output_shapes
:
*
dtype0
!dense_1/kernel/Regularizer/L2LossL2Loss8dense_1/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: e
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0*dense_1/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: n
IdentityIdentityquant_dense_2/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=
NoOpNoOp0^conv2d/kernel/Regularizer/L2Loss/ReadVariableOp2^conv2d_1/kernel/Regularizer/L2Loss/ReadVariableOp2^conv2d_2/kernel/Regularizer/L2Loss/ReadVariableOp2^conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOp2^conv2d_4/kernel/Regularizer/L2Loss/ReadVariableOp/^dense/kernel/Regularizer/L2Loss/ReadVariableOp1^dense_1/kernel/Regularizer/L2Loss/ReadVariableOp$^quant_conv2d/BiasAdd/ReadVariableOp*^quant_conv2d/LastValueQuant/AssignMaxLast*^quant_conv2d/LastValueQuant/AssignMinLast4^quant_conv2d/LastValueQuant/BatchMax/ReadVariableOp4^quant_conv2d/LastValueQuant/BatchMin/ReadVariableOpM^quant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpO^quant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1O^quant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2@^quant_conv2d/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp;^quant_conv2d/MovingAvgQuantize/AssignMaxEma/ReadVariableOp@^quant_conv2d/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp;^quant_conv2d/MovingAvgQuantize/AssignMinEma/ReadVariableOpF^quant_conv2d/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpH^quant_conv2d/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1&^quant_conv2d_1/BiasAdd/ReadVariableOp,^quant_conv2d_1/LastValueQuant/AssignMaxLast,^quant_conv2d_1/LastValueQuant/AssignMinLast6^quant_conv2d_1/LastValueQuant/BatchMax/ReadVariableOp6^quant_conv2d_1/LastValueQuant/BatchMin/ReadVariableOpO^quant_conv2d_1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpQ^quant_conv2d_1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1Q^quant_conv2d_1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2B^quant_conv2d_1/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp=^quant_conv2d_1/MovingAvgQuantize/AssignMaxEma/ReadVariableOpB^quant_conv2d_1/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp=^quant_conv2d_1/MovingAvgQuantize/AssignMinEma/ReadVariableOpH^quant_conv2d_1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpJ^quant_conv2d_1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1&^quant_conv2d_2/BiasAdd/ReadVariableOp,^quant_conv2d_2/LastValueQuant/AssignMaxLast,^quant_conv2d_2/LastValueQuant/AssignMinLast6^quant_conv2d_2/LastValueQuant/BatchMax/ReadVariableOp6^quant_conv2d_2/LastValueQuant/BatchMin/ReadVariableOpO^quant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpQ^quant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1Q^quant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2B^quant_conv2d_2/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp=^quant_conv2d_2/MovingAvgQuantize/AssignMaxEma/ReadVariableOpB^quant_conv2d_2/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp=^quant_conv2d_2/MovingAvgQuantize/AssignMinEma/ReadVariableOpH^quant_conv2d_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpJ^quant_conv2d_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1&^quant_conv2d_3/BiasAdd/ReadVariableOp,^quant_conv2d_3/LastValueQuant/AssignMaxLast,^quant_conv2d_3/LastValueQuant/AssignMinLast6^quant_conv2d_3/LastValueQuant/BatchMax/ReadVariableOp6^quant_conv2d_3/LastValueQuant/BatchMin/ReadVariableOpO^quant_conv2d_3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpQ^quant_conv2d_3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1Q^quant_conv2d_3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2B^quant_conv2d_3/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp=^quant_conv2d_3/MovingAvgQuantize/AssignMaxEma/ReadVariableOpB^quant_conv2d_3/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp=^quant_conv2d_3/MovingAvgQuantize/AssignMinEma/ReadVariableOpH^quant_conv2d_3/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpJ^quant_conv2d_3/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1&^quant_conv2d_4/BiasAdd/ReadVariableOp,^quant_conv2d_4/LastValueQuant/AssignMaxLast,^quant_conv2d_4/LastValueQuant/AssignMinLast6^quant_conv2d_4/LastValueQuant/BatchMax/ReadVariableOp6^quant_conv2d_4/LastValueQuant/BatchMin/ReadVariableOpO^quant_conv2d_4/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpQ^quant_conv2d_4/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1Q^quant_conv2d_4/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2B^quant_conv2d_4/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp=^quant_conv2d_4/MovingAvgQuantize/AssignMaxEma/ReadVariableOpB^quant_conv2d_4/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp=^quant_conv2d_4/MovingAvgQuantize/AssignMinEma/ReadVariableOpH^quant_conv2d_4/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpJ^quant_conv2d_4/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1#^quant_dense/BiasAdd/ReadVariableOp)^quant_dense/LastValueQuant/AssignMaxLast)^quant_dense/LastValueQuant/AssignMinLast3^quant_dense/LastValueQuant/BatchMax/ReadVariableOp3^quant_dense/LastValueQuant/BatchMin/ReadVariableOpB^quant_dense/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpD^quant_dense/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1D^quant_dense/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2?^quant_dense/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp:^quant_dense/MovingAvgQuantize/AssignMaxEma/ReadVariableOp?^quant_dense/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp:^quant_dense/MovingAvgQuantize/AssignMinEma/ReadVariableOpE^quant_dense/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpG^quant_dense/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1%^quant_dense_1/BiasAdd/ReadVariableOp+^quant_dense_1/LastValueQuant/AssignMaxLast+^quant_dense_1/LastValueQuant/AssignMinLast5^quant_dense_1/LastValueQuant/BatchMax/ReadVariableOp5^quant_dense_1/LastValueQuant/BatchMin/ReadVariableOpD^quant_dense_1/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpF^quant_dense_1/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1F^quant_dense_1/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2A^quant_dense_1/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp<^quant_dense_1/MovingAvgQuantize/AssignMaxEma/ReadVariableOpA^quant_dense_1/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp<^quant_dense_1/MovingAvgQuantize/AssignMinEma/ReadVariableOpG^quant_dense_1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpI^quant_dense_1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1%^quant_dense_2/BiasAdd/ReadVariableOp+^quant_dense_2/LastValueQuant/AssignMaxLast+^quant_dense_2/LastValueQuant/AssignMinLast5^quant_dense_2/LastValueQuant/BatchMax/ReadVariableOp5^quant_dense_2/LastValueQuant/BatchMin/ReadVariableOpD^quant_dense_2/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpF^quant_dense_2/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1F^quant_dense_2/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2A^quant_dense_2/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp<^quant_dense_2/MovingAvgQuantize/AssignMaxEma/ReadVariableOpA^quant_dense_2/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp<^quant_dense_2/MovingAvgQuantize/AssignMinEma/ReadVariableOpG^quant_dense_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpI^quant_dense_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_13^quantize_layer/AllValuesQuantize/AssignMaxAllValue3^quantize_layer/AllValuesQuantize/AssignMinAllValueH^quantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOpJ^quantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_18^quantize_layer/AllValuesQuantize/Maximum/ReadVariableOp8^quantize_layer/AllValuesQuantize/Minimum/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:ÿÿÿÿÿÿÿÿÿææ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2b
/conv2d/kernel/Regularizer/L2Loss/ReadVariableOp/conv2d/kernel/Regularizer/L2Loss/ReadVariableOp2f
1conv2d_1/kernel/Regularizer/L2Loss/ReadVariableOp1conv2d_1/kernel/Regularizer/L2Loss/ReadVariableOp2f
1conv2d_2/kernel/Regularizer/L2Loss/ReadVariableOp1conv2d_2/kernel/Regularizer/L2Loss/ReadVariableOp2f
1conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOp1conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOp2f
1conv2d_4/kernel/Regularizer/L2Loss/ReadVariableOp1conv2d_4/kernel/Regularizer/L2Loss/ReadVariableOp2`
.dense/kernel/Regularizer/L2Loss/ReadVariableOp.dense/kernel/Regularizer/L2Loss/ReadVariableOp2d
0dense_1/kernel/Regularizer/L2Loss/ReadVariableOp0dense_1/kernel/Regularizer/L2Loss/ReadVariableOp2J
#quant_conv2d/BiasAdd/ReadVariableOp#quant_conv2d/BiasAdd/ReadVariableOp2V
)quant_conv2d/LastValueQuant/AssignMaxLast)quant_conv2d/LastValueQuant/AssignMaxLast2V
)quant_conv2d/LastValueQuant/AssignMinLast)quant_conv2d/LastValueQuant/AssignMinLast2j
3quant_conv2d/LastValueQuant/BatchMax/ReadVariableOp3quant_conv2d/LastValueQuant/BatchMax/ReadVariableOp2j
3quant_conv2d/LastValueQuant/BatchMin/ReadVariableOp3quant_conv2d/LastValueQuant/BatchMin/ReadVariableOp2
Lquant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpLquant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp2 
Nquant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1Nquant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_12 
Nquant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2Nquant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_22
?quant_conv2d/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp?quant_conv2d/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2x
:quant_conv2d/MovingAvgQuantize/AssignMaxEma/ReadVariableOp:quant_conv2d/MovingAvgQuantize/AssignMaxEma/ReadVariableOp2
?quant_conv2d/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp?quant_conv2d/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2x
:quant_conv2d/MovingAvgQuantize/AssignMinEma/ReadVariableOp:quant_conv2d/MovingAvgQuantize/AssignMinEma/ReadVariableOp2
Equant_conv2d/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpEquant_conv2d/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2
Gquant_conv2d/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Gquant_conv2d/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12N
%quant_conv2d_1/BiasAdd/ReadVariableOp%quant_conv2d_1/BiasAdd/ReadVariableOp2Z
+quant_conv2d_1/LastValueQuant/AssignMaxLast+quant_conv2d_1/LastValueQuant/AssignMaxLast2Z
+quant_conv2d_1/LastValueQuant/AssignMinLast+quant_conv2d_1/LastValueQuant/AssignMinLast2n
5quant_conv2d_1/LastValueQuant/BatchMax/ReadVariableOp5quant_conv2d_1/LastValueQuant/BatchMax/ReadVariableOp2n
5quant_conv2d_1/LastValueQuant/BatchMin/ReadVariableOp5quant_conv2d_1/LastValueQuant/BatchMin/ReadVariableOp2 
Nquant_conv2d_1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpNquant_conv2d_1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp2¤
Pquant_conv2d_1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1Pquant_conv2d_1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_12¤
Pquant_conv2d_1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2Pquant_conv2d_1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_22
Aquant_conv2d_1/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpAquant_conv2d_1/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2|
<quant_conv2d_1/MovingAvgQuantize/AssignMaxEma/ReadVariableOp<quant_conv2d_1/MovingAvgQuantize/AssignMaxEma/ReadVariableOp2
Aquant_conv2d_1/MovingAvgQuantize/AssignMinEma/AssignSubVariableOpAquant_conv2d_1/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2|
<quant_conv2d_1/MovingAvgQuantize/AssignMinEma/ReadVariableOp<quant_conv2d_1/MovingAvgQuantize/AssignMinEma/ReadVariableOp2
Gquant_conv2d_1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpGquant_conv2d_1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2
Iquant_conv2d_1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Iquant_conv2d_1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12N
%quant_conv2d_2/BiasAdd/ReadVariableOp%quant_conv2d_2/BiasAdd/ReadVariableOp2Z
+quant_conv2d_2/LastValueQuant/AssignMaxLast+quant_conv2d_2/LastValueQuant/AssignMaxLast2Z
+quant_conv2d_2/LastValueQuant/AssignMinLast+quant_conv2d_2/LastValueQuant/AssignMinLast2n
5quant_conv2d_2/LastValueQuant/BatchMax/ReadVariableOp5quant_conv2d_2/LastValueQuant/BatchMax/ReadVariableOp2n
5quant_conv2d_2/LastValueQuant/BatchMin/ReadVariableOp5quant_conv2d_2/LastValueQuant/BatchMin/ReadVariableOp2 
Nquant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpNquant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp2¤
Pquant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1Pquant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_12¤
Pquant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2Pquant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_22
Aquant_conv2d_2/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpAquant_conv2d_2/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2|
<quant_conv2d_2/MovingAvgQuantize/AssignMaxEma/ReadVariableOp<quant_conv2d_2/MovingAvgQuantize/AssignMaxEma/ReadVariableOp2
Aquant_conv2d_2/MovingAvgQuantize/AssignMinEma/AssignSubVariableOpAquant_conv2d_2/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2|
<quant_conv2d_2/MovingAvgQuantize/AssignMinEma/ReadVariableOp<quant_conv2d_2/MovingAvgQuantize/AssignMinEma/ReadVariableOp2
Gquant_conv2d_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpGquant_conv2d_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2
Iquant_conv2d_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Iquant_conv2d_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12N
%quant_conv2d_3/BiasAdd/ReadVariableOp%quant_conv2d_3/BiasAdd/ReadVariableOp2Z
+quant_conv2d_3/LastValueQuant/AssignMaxLast+quant_conv2d_3/LastValueQuant/AssignMaxLast2Z
+quant_conv2d_3/LastValueQuant/AssignMinLast+quant_conv2d_3/LastValueQuant/AssignMinLast2n
5quant_conv2d_3/LastValueQuant/BatchMax/ReadVariableOp5quant_conv2d_3/LastValueQuant/BatchMax/ReadVariableOp2n
5quant_conv2d_3/LastValueQuant/BatchMin/ReadVariableOp5quant_conv2d_3/LastValueQuant/BatchMin/ReadVariableOp2 
Nquant_conv2d_3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpNquant_conv2d_3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp2¤
Pquant_conv2d_3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1Pquant_conv2d_3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_12¤
Pquant_conv2d_3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2Pquant_conv2d_3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_22
Aquant_conv2d_3/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpAquant_conv2d_3/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2|
<quant_conv2d_3/MovingAvgQuantize/AssignMaxEma/ReadVariableOp<quant_conv2d_3/MovingAvgQuantize/AssignMaxEma/ReadVariableOp2
Aquant_conv2d_3/MovingAvgQuantize/AssignMinEma/AssignSubVariableOpAquant_conv2d_3/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2|
<quant_conv2d_3/MovingAvgQuantize/AssignMinEma/ReadVariableOp<quant_conv2d_3/MovingAvgQuantize/AssignMinEma/ReadVariableOp2
Gquant_conv2d_3/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpGquant_conv2d_3/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2
Iquant_conv2d_3/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Iquant_conv2d_3/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12N
%quant_conv2d_4/BiasAdd/ReadVariableOp%quant_conv2d_4/BiasAdd/ReadVariableOp2Z
+quant_conv2d_4/LastValueQuant/AssignMaxLast+quant_conv2d_4/LastValueQuant/AssignMaxLast2Z
+quant_conv2d_4/LastValueQuant/AssignMinLast+quant_conv2d_4/LastValueQuant/AssignMinLast2n
5quant_conv2d_4/LastValueQuant/BatchMax/ReadVariableOp5quant_conv2d_4/LastValueQuant/BatchMax/ReadVariableOp2n
5quant_conv2d_4/LastValueQuant/BatchMin/ReadVariableOp5quant_conv2d_4/LastValueQuant/BatchMin/ReadVariableOp2 
Nquant_conv2d_4/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpNquant_conv2d_4/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp2¤
Pquant_conv2d_4/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1Pquant_conv2d_4/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_12¤
Pquant_conv2d_4/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2Pquant_conv2d_4/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_22
Aquant_conv2d_4/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpAquant_conv2d_4/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2|
<quant_conv2d_4/MovingAvgQuantize/AssignMaxEma/ReadVariableOp<quant_conv2d_4/MovingAvgQuantize/AssignMaxEma/ReadVariableOp2
Aquant_conv2d_4/MovingAvgQuantize/AssignMinEma/AssignSubVariableOpAquant_conv2d_4/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2|
<quant_conv2d_4/MovingAvgQuantize/AssignMinEma/ReadVariableOp<quant_conv2d_4/MovingAvgQuantize/AssignMinEma/ReadVariableOp2
Gquant_conv2d_4/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpGquant_conv2d_4/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2
Iquant_conv2d_4/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Iquant_conv2d_4/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12H
"quant_dense/BiasAdd/ReadVariableOp"quant_dense/BiasAdd/ReadVariableOp2T
(quant_dense/LastValueQuant/AssignMaxLast(quant_dense/LastValueQuant/AssignMaxLast2T
(quant_dense/LastValueQuant/AssignMinLast(quant_dense/LastValueQuant/AssignMinLast2h
2quant_dense/LastValueQuant/BatchMax/ReadVariableOp2quant_dense/LastValueQuant/BatchMax/ReadVariableOp2h
2quant_dense/LastValueQuant/BatchMin/ReadVariableOp2quant_dense/LastValueQuant/BatchMin/ReadVariableOp2
Aquant_dense/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpAquant_dense/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp2
Cquant_dense/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1Cquant_dense/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_12
Cquant_dense/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2Cquant_dense/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_22
>quant_dense/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp>quant_dense/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2v
9quant_dense/MovingAvgQuantize/AssignMaxEma/ReadVariableOp9quant_dense/MovingAvgQuantize/AssignMaxEma/ReadVariableOp2
>quant_dense/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp>quant_dense/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2v
9quant_dense/MovingAvgQuantize/AssignMinEma/ReadVariableOp9quant_dense/MovingAvgQuantize/AssignMinEma/ReadVariableOp2
Dquant_dense/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpDquant_dense/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2
Fquant_dense/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Fquant_dense/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12L
$quant_dense_1/BiasAdd/ReadVariableOp$quant_dense_1/BiasAdd/ReadVariableOp2X
*quant_dense_1/LastValueQuant/AssignMaxLast*quant_dense_1/LastValueQuant/AssignMaxLast2X
*quant_dense_1/LastValueQuant/AssignMinLast*quant_dense_1/LastValueQuant/AssignMinLast2l
4quant_dense_1/LastValueQuant/BatchMax/ReadVariableOp4quant_dense_1/LastValueQuant/BatchMax/ReadVariableOp2l
4quant_dense_1/LastValueQuant/BatchMin/ReadVariableOp4quant_dense_1/LastValueQuant/BatchMin/ReadVariableOp2
Cquant_dense_1/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpCquant_dense_1/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp2
Equant_dense_1/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1Equant_dense_1/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_12
Equant_dense_1/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2Equant_dense_1/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_22
@quant_dense_1/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp@quant_dense_1/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2z
;quant_dense_1/MovingAvgQuantize/AssignMaxEma/ReadVariableOp;quant_dense_1/MovingAvgQuantize/AssignMaxEma/ReadVariableOp2
@quant_dense_1/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp@quant_dense_1/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2z
;quant_dense_1/MovingAvgQuantize/AssignMinEma/ReadVariableOp;quant_dense_1/MovingAvgQuantize/AssignMinEma/ReadVariableOp2
Fquant_dense_1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpFquant_dense_1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2
Hquant_dense_1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Hquant_dense_1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12L
$quant_dense_2/BiasAdd/ReadVariableOp$quant_dense_2/BiasAdd/ReadVariableOp2X
*quant_dense_2/LastValueQuant/AssignMaxLast*quant_dense_2/LastValueQuant/AssignMaxLast2X
*quant_dense_2/LastValueQuant/AssignMinLast*quant_dense_2/LastValueQuant/AssignMinLast2l
4quant_dense_2/LastValueQuant/BatchMax/ReadVariableOp4quant_dense_2/LastValueQuant/BatchMax/ReadVariableOp2l
4quant_dense_2/LastValueQuant/BatchMin/ReadVariableOp4quant_dense_2/LastValueQuant/BatchMin/ReadVariableOp2
Cquant_dense_2/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpCquant_dense_2/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp2
Equant_dense_2/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1Equant_dense_2/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_12
Equant_dense_2/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2Equant_dense_2/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_22
@quant_dense_2/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp@quant_dense_2/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2z
;quant_dense_2/MovingAvgQuantize/AssignMaxEma/ReadVariableOp;quant_dense_2/MovingAvgQuantize/AssignMaxEma/ReadVariableOp2
@quant_dense_2/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp@quant_dense_2/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2z
;quant_dense_2/MovingAvgQuantize/AssignMinEma/ReadVariableOp;quant_dense_2/MovingAvgQuantize/AssignMinEma/ReadVariableOp2
Fquant_dense_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpFquant_dense_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2
Hquant_dense_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Hquant_dense_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12h
2quantize_layer/AllValuesQuantize/AssignMaxAllValue2quantize_layer/AllValuesQuantize/AssignMaxAllValue2h
2quantize_layer/AllValuesQuantize/AssignMinAllValue2quantize_layer/AllValuesQuantize/AssignMinAllValue2
Gquantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOpGquantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2
Iquantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Iquantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12r
7quantize_layer/AllValuesQuantize/Maximum/ReadVariableOp7quantize_layer/AllValuesQuantize/Maximum/ReadVariableOp2r
7quantize_layer/AllValuesQuantize/Minimum/ReadVariableOp7quantize_layer/AllValuesQuantize/Minimum/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿææ
 
_user_specified_nameinputs
ÿ	
h
I__inference_quant_dropout_layer_call_and_return_conditional_losses_143654

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?§
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿp
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ì

F__inference_sequential_layer_call_and_return_conditional_losses_145046
conv2d_input
quantize_layer_144902: 
quantize_layer_144904: -
quant_conv2d_144907: !
quant_conv2d_144909: !
quant_conv2d_144911: !
quant_conv2d_144913: 
quant_conv2d_144915: 
quant_conv2d_144917: /
quant_conv2d_1_144921:  #
quant_conv2d_1_144923: #
quant_conv2d_1_144925: #
quant_conv2d_1_144927: 
quant_conv2d_1_144929: 
quant_conv2d_1_144931: /
quant_conv2d_2_144935: @#
quant_conv2d_2_144937:@#
quant_conv2d_2_144939:@#
quant_conv2d_2_144941:@
quant_conv2d_2_144943: 
quant_conv2d_2_144945: /
quant_conv2d_3_144948:@@#
quant_conv2d_3_144950:@#
quant_conv2d_3_144952:@#
quant_conv2d_3_144954:@
quant_conv2d_3_144956: 
quant_conv2d_3_144958: /
quant_conv2d_4_144961:@ #
quant_conv2d_4_144963: #
quant_conv2d_4_144965: #
quant_conv2d_4_144967: 
quant_conv2d_4_144969: 
quant_conv2d_4_144971: &
quant_dense_144976:

quant_dense_144978: 
quant_dense_144980: !
quant_dense_144982:	
quant_dense_144984: 
quant_dense_144986: (
quant_dense_1_144990:

quant_dense_1_144992: 
quant_dense_1_144994: #
quant_dense_1_144996:	
quant_dense_1_144998: 
quant_dense_1_145000: '
quant_dense_2_145004:	
quant_dense_2_145006: 
quant_dense_2_145008: "
quant_dense_2_145010:
quant_dense_2_145012: 
quant_dense_2_145014: 
identity¢/conv2d/kernel/Regularizer/L2Loss/ReadVariableOp¢1conv2d_1/kernel/Regularizer/L2Loss/ReadVariableOp¢1conv2d_2/kernel/Regularizer/L2Loss/ReadVariableOp¢1conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOp¢1conv2d_4/kernel/Regularizer/L2Loss/ReadVariableOp¢.dense/kernel/Regularizer/L2Loss/ReadVariableOp¢0dense_1/kernel/Regularizer/L2Loss/ReadVariableOp¢$quant_conv2d/StatefulPartitionedCall¢&quant_conv2d_1/StatefulPartitionedCall¢&quant_conv2d_2/StatefulPartitionedCall¢&quant_conv2d_3/StatefulPartitionedCall¢&quant_conv2d_4/StatefulPartitionedCall¢#quant_dense/StatefulPartitionedCall¢%quant_dense_1/StatefulPartitionedCall¢%quant_dense_2/StatefulPartitionedCall¢%quant_dropout/StatefulPartitionedCall¢'quant_dropout_1/StatefulPartitionedCall¢&quantize_layer/StatefulPartitionedCall¢
&quantize_layer/StatefulPartitionedCallStatefulPartitionedCallconv2d_inputquantize_layer_144902quantize_layer_144904*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿææ* 
_read_only_resource_inputs
 *;
config_proto+)

CPU

GPU2*0J
	
   E 8 *S
fNRL
J__inference_quantize_layer_layer_call_and_return_conditional_losses_144283
$quant_conv2d/StatefulPartitionedCallStatefulPartitionedCall/quantize_layer/StatefulPartitionedCall:output:0quant_conv2d_144907quant_conv2d_144909quant_conv2d_144911quant_conv2d_144913quant_conv2d_144915quant_conv2d_144917*
Tin
	2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿââ *$
_read_only_resource_inputs
*;
config_proto+)

CPU

GPU2*0J
	
   E 8 *Q
fLRJ
H__inference_quant_conv2d_layer_call_and_return_conditional_losses_144235
#quant_max_pooling2d/PartitionedCallPartitionedCall-quant_conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿKK * 
_read_only_resource_inputs
 *;
config_proto+)

CPU

GPU2*0J
	
   E 8 *X
fSRQ
O__inference_quant_max_pooling2d_layer_call_and_return_conditional_losses_144159¤
&quant_conv2d_1/StatefulPartitionedCallStatefulPartitionedCall,quant_max_pooling2d/PartitionedCall:output:0quant_conv2d_1_144921quant_conv2d_1_144923quant_conv2d_1_144925quant_conv2d_1_144927quant_conv2d_1_144929quant_conv2d_1_144931*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿKK *$
_read_only_resource_inputs
*;
config_proto+)

CPU

GPU2*0J
	
   E 8 *S
fNRL
J__inference_quant_conv2d_1_layer_call_and_return_conditional_losses_144131
%quant_max_pooling2d_1/PartitionedCallPartitionedCall/quant_conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *;
config_proto+)

CPU

GPU2*0J
	
   E 8 *Z
fURS
Q__inference_quant_max_pooling2d_1_layer_call_and_return_conditional_losses_144055¦
&quant_conv2d_2/StatefulPartitionedCallStatefulPartitionedCall.quant_max_pooling2d_1/PartitionedCall:output:0quant_conv2d_2_144935quant_conv2d_2_144937quant_conv2d_2_144939quant_conv2d_2_144941quant_conv2d_2_144943quant_conv2d_2_144945*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*;
config_proto+)

CPU

GPU2*0J
	
   E 8 *S
fNRL
J__inference_quant_conv2d_2_layer_call_and_return_conditional_losses_144027§
&quant_conv2d_3/StatefulPartitionedCallStatefulPartitionedCall/quant_conv2d_2/StatefulPartitionedCall:output:0quant_conv2d_3_144948quant_conv2d_3_144950quant_conv2d_3_144952quant_conv2d_3_144954quant_conv2d_3_144956quant_conv2d_3_144958*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*;
config_proto+)

CPU

GPU2*0J
	
   E 8 *S
fNRL
J__inference_quant_conv2d_3_layer_call_and_return_conditional_losses_143939§
&quant_conv2d_4/StatefulPartitionedCallStatefulPartitionedCall/quant_conv2d_3/StatefulPartitionedCall:output:0quant_conv2d_4_144961quant_conv2d_4_144963quant_conv2d_4_144965quant_conv2d_4_144967quant_conv2d_4_144969quant_conv2d_4_144971*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*;
config_proto+)

CPU

GPU2*0J
	
   E 8 *S
fNRL
J__inference_quant_conv2d_4_layer_call_and_return_conditional_losses_143851
%quant_max_pooling2d_2/PartitionedCallPartitionedCall/quant_conv2d_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *;
config_proto+)

CPU

GPU2*0J
	
   E 8 *Z
fURS
Q__inference_quant_max_pooling2d_2_layer_call_and_return_conditional_losses_143775ù
quant_flatten/PartitionedCallPartitionedCall.quant_max_pooling2d_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *;
config_proto+)

CPU

GPU2*0J
	
   E 8 *R
fMRK
I__inference_quant_flatten_layer_call_and_return_conditional_losses_143759ÿ
#quant_dense/StatefulPartitionedCallStatefulPartitionedCall&quant_flatten/PartitionedCall:output:0quant_dense_144976quant_dense_144978quant_dense_144980quant_dense_144982quant_dense_144984quant_dense_144986*
Tin
	2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*;
config_proto+)

CPU

GPU2*0J
	
   E 8 *P
fKRI
G__inference_quant_dense_layer_call_and_return_conditional_losses_143730
%quant_dropout/StatefulPartitionedCallStatefulPartitionedCall,quant_dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *;
config_proto+)

CPU

GPU2*0J
	
   E 8 *R
fMRK
I__inference_quant_dropout_layer_call_and_return_conditional_losses_143654
%quant_dense_1/StatefulPartitionedCallStatefulPartitionedCall.quant_dropout/StatefulPartitionedCall:output:0quant_dense_1_144990quant_dense_1_144992quant_dense_1_144994quant_dense_1_144996quant_dense_1_144998quant_dense_1_145000*
Tin
	2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*;
config_proto+)

CPU

GPU2*0J
	
   E 8 *R
fMRK
I__inference_quant_dense_1_layer_call_and_return_conditional_losses_143619µ
'quant_dropout_1/StatefulPartitionedCallStatefulPartitionedCall.quant_dense_1/StatefulPartitionedCall:output:0&^quant_dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *;
config_proto+)

CPU

GPU2*0J
	
   E 8 *T
fORM
K__inference_quant_dropout_1_layer_call_and_return_conditional_losses_143543
%quant_dense_2/StatefulPartitionedCallStatefulPartitionedCall0quant_dropout_1/StatefulPartitionedCall:output:0quant_dense_2_145004quant_dense_2_145006quant_dense_2_145008quant_dense_2_145010quant_dense_2_145012quant_dense_2_145014*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*;
config_proto+)

CPU

GPU2*0J
	
   E 8 *R
fMRK
I__inference_quant_dense_2_layer_call_and_return_conditional_losses_143508
/conv2d/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpquant_conv2d_144907*&
_output_shapes
: *
dtype0
 conv2d/kernel/Regularizer/L2LossL2Loss7conv2d/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: d
conv2d/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;
conv2d/kernel/Regularizer/mulMul(conv2d/kernel/Regularizer/mul/x:output:0)conv2d/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: 
1conv2d_1/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpquant_conv2d_1_144921*&
_output_shapes
:  *
dtype0
"conv2d_1/kernel/Regularizer/L2LossL2Loss9conv2d_1/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!conv2d_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o; 
conv2d_1/kernel/Regularizer/mulMul*conv2d_1/kernel/Regularizer/mul/x:output:0+conv2d_1/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: 
1conv2d_2/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpquant_conv2d_2_144935*&
_output_shapes
: @*
dtype0
"conv2d_2/kernel/Regularizer/L2LossL2Loss9conv2d_2/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!conv2d_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o; 
conv2d_2/kernel/Regularizer/mulMul*conv2d_2/kernel/Regularizer/mul/x:output:0+conv2d_2/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: 
1conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpquant_conv2d_3_144948*&
_output_shapes
:@@*
dtype0
"conv2d_3/kernel/Regularizer/L2LossL2Loss9conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!conv2d_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o; 
conv2d_3/kernel/Regularizer/mulMul*conv2d_3/kernel/Regularizer/mul/x:output:0+conv2d_3/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: 
1conv2d_4/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpquant_conv2d_4_144961*&
_output_shapes
:@ *
dtype0
"conv2d_4/kernel/Regularizer/L2LossL2Loss9conv2d_4/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!conv2d_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o; 
conv2d_4/kernel/Regularizer/mulMul*conv2d_4/kernel/Regularizer/mul/x:output:0+conv2d_4/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: 
.dense/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpquant_dense_144976* 
_output_shapes
:
*
dtype0
dense/kernel/Regularizer/L2LossL2Loss6dense/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: c
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0(dense/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: 
0dense_1/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpquant_dense_1_144990* 
_output_shapes
:
*
dtype0
!dense_1/kernel/Regularizer/L2LossL2Loss8dense_1/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: e
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0*dense_1/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: }
IdentityIdentity.quant_dense_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿè
NoOpNoOp0^conv2d/kernel/Regularizer/L2Loss/ReadVariableOp2^conv2d_1/kernel/Regularizer/L2Loss/ReadVariableOp2^conv2d_2/kernel/Regularizer/L2Loss/ReadVariableOp2^conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOp2^conv2d_4/kernel/Regularizer/L2Loss/ReadVariableOp/^dense/kernel/Regularizer/L2Loss/ReadVariableOp1^dense_1/kernel/Regularizer/L2Loss/ReadVariableOp%^quant_conv2d/StatefulPartitionedCall'^quant_conv2d_1/StatefulPartitionedCall'^quant_conv2d_2/StatefulPartitionedCall'^quant_conv2d_3/StatefulPartitionedCall'^quant_conv2d_4/StatefulPartitionedCall$^quant_dense/StatefulPartitionedCall&^quant_dense_1/StatefulPartitionedCall&^quant_dense_2/StatefulPartitionedCall&^quant_dropout/StatefulPartitionedCall(^quant_dropout_1/StatefulPartitionedCall'^quantize_layer/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:ÿÿÿÿÿÿÿÿÿææ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2b
/conv2d/kernel/Regularizer/L2Loss/ReadVariableOp/conv2d/kernel/Regularizer/L2Loss/ReadVariableOp2f
1conv2d_1/kernel/Regularizer/L2Loss/ReadVariableOp1conv2d_1/kernel/Regularizer/L2Loss/ReadVariableOp2f
1conv2d_2/kernel/Regularizer/L2Loss/ReadVariableOp1conv2d_2/kernel/Regularizer/L2Loss/ReadVariableOp2f
1conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOp1conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOp2f
1conv2d_4/kernel/Regularizer/L2Loss/ReadVariableOp1conv2d_4/kernel/Regularizer/L2Loss/ReadVariableOp2`
.dense/kernel/Regularizer/L2Loss/ReadVariableOp.dense/kernel/Regularizer/L2Loss/ReadVariableOp2d
0dense_1/kernel/Regularizer/L2Loss/ReadVariableOp0dense_1/kernel/Regularizer/L2Loss/ReadVariableOp2L
$quant_conv2d/StatefulPartitionedCall$quant_conv2d/StatefulPartitionedCall2P
&quant_conv2d_1/StatefulPartitionedCall&quant_conv2d_1/StatefulPartitionedCall2P
&quant_conv2d_2/StatefulPartitionedCall&quant_conv2d_2/StatefulPartitionedCall2P
&quant_conv2d_3/StatefulPartitionedCall&quant_conv2d_3/StatefulPartitionedCall2P
&quant_conv2d_4/StatefulPartitionedCall&quant_conv2d_4/StatefulPartitionedCall2J
#quant_dense/StatefulPartitionedCall#quant_dense/StatefulPartitionedCall2N
%quant_dense_1/StatefulPartitionedCall%quant_dense_1/StatefulPartitionedCall2N
%quant_dense_2/StatefulPartitionedCall%quant_dense_2/StatefulPartitionedCall2N
%quant_dropout/StatefulPartitionedCall%quant_dropout/StatefulPartitionedCall2R
'quant_dropout_1/StatefulPartitionedCall'quant_dropout_1/StatefulPartitionedCall2P
&quantize_layer/StatefulPartitionedCall&quantize_layer/StatefulPartitionedCall:_ [
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿææ
&
_user_specified_nameconv2d_input
é
ð
.__inference_quant_dense_2_layer_call_fn_147065

inputs
unknown:	
	unknown_0: 
	unknown_1: 
	unknown_2:
	unknown_3: 
	unknown_4: 
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*;
config_proto+)

CPU

GPU2*0J
	
   E 8 *R
fMRK
I__inference_quant_dense_2_layer_call_and_return_conditional_losses_143508o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¤
Ú

+__inference_sequential_layer_call_fn_145320

inputs
unknown: 
	unknown_0: #
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: 
	unknown_5: 
	unknown_6: #
	unknown_7:  
	unknown_8: 
	unknown_9: 

unknown_10: 

unknown_11: 

unknown_12: $

unknown_13: @

unknown_14:@

unknown_15:@

unknown_16:@

unknown_17: 

unknown_18: $

unknown_19:@@

unknown_20:@

unknown_21:@

unknown_22:@

unknown_23: 

unknown_24: $

unknown_25:@ 

unknown_26: 

unknown_27: 

unknown_28: 

unknown_29: 

unknown_30: 

unknown_31:


unknown_32: 

unknown_33: 

unknown_34:	

unknown_35: 

unknown_36: 

unknown_37:


unknown_38: 

unknown_39: 

unknown_40:	

unknown_41: 

unknown_42: 

unknown_43:	

unknown_44: 

unknown_45: 

unknown_46:

unknown_47: 

unknown_48: 
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48*>
Tin7
523*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*T
_read_only_resource_inputs6
42	
 !"#$%&'()*+,-./012*;
config_proto+)

CPU

GPU2*0J
	
   E 8 *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_143336o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:ÿÿÿÿÿÿÿÿÿææ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿææ
 
_user_specified_nameinputs
í
ð
,__inference_quant_dense_layer_call_fn_146762

inputs
unknown:

	unknown_0: 
	unknown_1: 
	unknown_2:	
	unknown_3: 
	unknown_4: 
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*;
config_proto+)

CPU

GPU2*0J
	
   E 8 *P
fKRI
G__inference_quant_dense_layer_call_and_return_conditional_losses_143205p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
­
m
Q__inference_quant_max_pooling2d_2_layer_call_and_return_conditional_losses_146723

inputs
identity
MaxPoolMaxPoolinputs*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
ksize
*
paddingVALID*
strides
`
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ :W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
í
ò
.__inference_quant_dense_1_layer_call_fn_146922

inputs
unknown:

	unknown_0: 
	unknown_1: 
	unknown_2:	
	unknown_3: 
	unknown_4: 
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*;
config_proto+)

CPU

GPU2*0J
	
   E 8 *R
fMRK
I__inference_quant_dense_1_layer_call_and_return_conditional_losses_143619p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ÀT
	
I__inference_quant_dense_1_layer_call_and_return_conditional_losses_147000

inputsC
/lastvaluequant_batchmin_readvariableop_resource:
/
%lastvaluequant_assignminlast_resource: /
%lastvaluequant_assignmaxlast_resource: .
biasadd_readvariableop_resource:	@
6movingavgquantize_assignminema_readvariableop_resource: @
6movingavgquantize_assignmaxema_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢LastValueQuant/AssignMaxLast¢LastValueQuant/AssignMinLast¢&LastValueQuant/BatchMax/ReadVariableOp¢&LastValueQuant/BatchMin/ReadVariableOp¢5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp¢7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1¢7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2¢2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp¢-MovingAvgQuantize/AssignMaxEma/ReadVariableOp¢2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp¢-MovingAvgQuantize/AssignMinEma/ReadVariableOp¢8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp¢:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1¢0dense_1/kernel/Regularizer/L2Loss/ReadVariableOpe
LastValueQuant/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
&LastValueQuant/BatchMin/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource* 
_output_shapes
:
*
dtype0
LastValueQuant/BatchMinMin.LastValueQuant/BatchMin/ReadVariableOp:value:0LastValueQuant/Const:output:0*
T0*
_output_shapes
: g
LastValueQuant/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       
&LastValueQuant/BatchMax/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource* 
_output_shapes
:
*
dtype0
LastValueQuant/BatchMaxMax.LastValueQuant/BatchMax/ReadVariableOp:value:0LastValueQuant/Const_1:output:0*
T0*
_output_shapes
: ]
LastValueQuant/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ¿
LastValueQuant/truedivRealDiv LastValueQuant/BatchMax:output:0!LastValueQuant/truediv/y:output:0*
T0*
_output_shapes
: 
LastValueQuant/MinimumMinimum LastValueQuant/BatchMin:output:0LastValueQuant/truediv:z:0*
T0*
_output_shapes
: Y
LastValueQuant/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ¿{
LastValueQuant/mulMul LastValueQuant/BatchMin:output:0LastValueQuant/mul/y:output:0*
T0*
_output_shapes
: |
LastValueQuant/MaximumMaximum LastValueQuant/BatchMax:output:0LastValueQuant/mul:z:0*
T0*
_output_shapes
: ¬
LastValueQuant/AssignMinLastAssignVariableOp%lastvaluequant_assignminlast_resourceLastValueQuant/Minimum:z:0*
_output_shapes
 *
dtype0*
validate_shape(¬
LastValueQuant/AssignMaxLastAssignVariableOp%lastvaluequant_assignmaxlast_resourceLastValueQuant/Maximum:z:0*
_output_shapes
 *
dtype0*
validate_shape(§
5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource* 
_output_shapes
:
*
dtype0´
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp%lastvaluequant_assignminlast_resource^LastValueQuant/AssignMinLast*
_output_shapes
: *
dtype0´
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2ReadVariableOp%lastvaluequant_assignmaxlast_resource^LastValueQuant/AssignMaxLast*
_output_shapes
: *
dtype0¸
&LastValueQuant/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars=LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp:value:0?LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0?LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2:value:0* 
_output_shapes
:
*
narrow_range(}
MatMulMatMulinputs0LastValueQuant/FakeQuantWithMinMaxVars:outputs:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
MovingAvgQuantize/ConstConst*
_output_shapes
:*
dtype0*
valueB"       x
MovingAvgQuantize/BatchMinMinRelu:activations:0 MovingAvgQuantize/Const:output:0*
T0*
_output_shapes
: j
MovingAvgQuantize/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       z
MovingAvgQuantize/BatchMaxMaxRelu:activations:0"MovingAvgQuantize/Const_1:output:0*
T0*
_output_shapes
: `
MovingAvgQuantize/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
MovingAvgQuantize/MinimumMinimum#MovingAvgQuantize/BatchMin:output:0$MovingAvgQuantize/Minimum/y:output:0*
T0*
_output_shapes
: `
MovingAvgQuantize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
MovingAvgQuantize/MaximumMaximum#MovingAvgQuantize/BatchMax:output:0$MovingAvgQuantize/Maximum/y:output:0*
T0*
_output_shapes
: i
$MovingAvgQuantize/AssignMinEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o:
-MovingAvgQuantize/AssignMinEma/ReadVariableOpReadVariableOp6movingavgquantize_assignminema_readvariableop_resource*
_output_shapes
: *
dtype0 
"MovingAvgQuantize/AssignMinEma/subSub5MovingAvgQuantize/AssignMinEma/ReadVariableOp:value:0MovingAvgQuantize/Minimum:z:0*
T0*
_output_shapes
: ¡
"MovingAvgQuantize/AssignMinEma/mulMul&MovingAvgQuantize/AssignMinEma/sub:z:0-MovingAvgQuantize/AssignMinEma/decay:output:0*
T0*
_output_shapes
: ü
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOpAssignSubVariableOp6movingavgquantize_assignminema_readvariableop_resource&MovingAvgQuantize/AssignMinEma/mul:z:0.^MovingAvgQuantize/AssignMinEma/ReadVariableOp*
_output_shapes
 *
dtype0i
$MovingAvgQuantize/AssignMaxEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o:
-MovingAvgQuantize/AssignMaxEma/ReadVariableOpReadVariableOp6movingavgquantize_assignmaxema_readvariableop_resource*
_output_shapes
: *
dtype0 
"MovingAvgQuantize/AssignMaxEma/subSub5MovingAvgQuantize/AssignMaxEma/ReadVariableOp:value:0MovingAvgQuantize/Maximum:z:0*
T0*
_output_shapes
: ¡
"MovingAvgQuantize/AssignMaxEma/mulMul&MovingAvgQuantize/AssignMaxEma/sub:z:0-MovingAvgQuantize/AssignMaxEma/decay:output:0*
T0*
_output_shapes
: ü
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpAssignSubVariableOp6movingavgquantize_assignmaxema_readvariableop_resource&MovingAvgQuantize/AssignMaxEma/mul:z:0.^MovingAvgQuantize/AssignMaxEma/ReadVariableOp*
_output_shapes
 *
dtype0Ü
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp6movingavgquantize_assignminema_readvariableop_resource3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp*
_output_shapes
: *
dtype0Þ
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp6movingavgquantize_assignmaxema_readvariableop_resource3^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp*
_output_shapes
: *
dtype0
)MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsRelu:activations:0@MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BMovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢
0dense_1/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource* 
_output_shapes
:
*
dtype0
!dense_1/kernel/Regularizer/L2LossL2Loss8dense_1/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: e
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0*dense_1/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: 
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp^BiasAdd/ReadVariableOp^LastValueQuant/AssignMaxLast^LastValueQuant/AssignMinLast'^LastValueQuant/BatchMax/ReadVariableOp'^LastValueQuant/BatchMin/ReadVariableOp6^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp8^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_18^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_23^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp.^MovingAvgQuantize/AssignMaxEma/ReadVariableOp3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp.^MovingAvgQuantize/AssignMinEma/ReadVariableOp9^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_11^dense_1/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ: : : : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2<
LastValueQuant/AssignMaxLastLastValueQuant/AssignMaxLast2<
LastValueQuant/AssignMinLastLastValueQuant/AssignMinLast2P
&LastValueQuant/BatchMax/ReadVariableOp&LastValueQuant/BatchMax/ReadVariableOp2P
&LastValueQuant/BatchMin/ReadVariableOp&LastValueQuant/BatchMin/ReadVariableOp2n
5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp2r
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_17LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_12r
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_27LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_22h
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2^
-MovingAvgQuantize/AssignMaxEma/ReadVariableOp-MovingAvgQuantize/AssignMaxEma/ReadVariableOp2h
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2^
-MovingAvgQuantize/AssignMinEma/ReadVariableOp-MovingAvgQuantize/AssignMinEma/ReadVariableOp2t
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12d
0dense_1/kernel/Regularizer/L2Loss/ReadVariableOp0dense_1/kernel/Regularizer/L2Loss/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ãî
úB
F__inference_sequential_layer_call_and_return_conditional_losses_145605

inputsZ
Pquantize_layer_allvaluesquantize_fakequantwithminmaxvars_readvariableop_resource: \
Rquantize_layer_allvaluesquantize_fakequantwithminmaxvars_readvariableop_1_resource: o
Uquant_conv2d_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource: e
Wquant_conv2d_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource: e
Wquant_conv2d_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource: :
,quant_conv2d_biasadd_readvariableop_resource: X
Nquant_conv2d_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource: Z
Pquant_conv2d_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: q
Wquant_conv2d_1_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource:  g
Yquant_conv2d_1_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource: g
Yquant_conv2d_1_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource: <
.quant_conv2d_1_biasadd_readvariableop_resource: Z
Pquant_conv2d_1_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource: \
Rquant_conv2d_1_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: q
Wquant_conv2d_2_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource: @g
Yquant_conv2d_2_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource:@g
Yquant_conv2d_2_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource:@<
.quant_conv2d_2_biasadd_readvariableop_resource:@Z
Pquant_conv2d_2_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource: \
Rquant_conv2d_2_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: q
Wquant_conv2d_3_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource:@@g
Yquant_conv2d_3_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource:@g
Yquant_conv2d_3_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource:@<
.quant_conv2d_3_biasadd_readvariableop_resource:@Z
Pquant_conv2d_3_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource: \
Rquant_conv2d_3_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: q
Wquant_conv2d_4_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource:@ g
Yquant_conv2d_4_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource: g
Yquant_conv2d_4_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource: <
.quant_conv2d_4_biasadd_readvariableop_resource: Z
Pquant_conv2d_4_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource: \
Rquant_conv2d_4_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: ^
Jquant_dense_lastvaluequant_fakequantwithminmaxvars_readvariableop_resource:
V
Lquant_dense_lastvaluequant_fakequantwithminmaxvars_readvariableop_1_resource: V
Lquant_dense_lastvaluequant_fakequantwithminmaxvars_readvariableop_2_resource: :
+quant_dense_biasadd_readvariableop_resource:	W
Mquant_dense_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource: Y
Oquant_dense_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: `
Lquant_dense_1_lastvaluequant_fakequantwithminmaxvars_readvariableop_resource:
X
Nquant_dense_1_lastvaluequant_fakequantwithminmaxvars_readvariableop_1_resource: X
Nquant_dense_1_lastvaluequant_fakequantwithminmaxvars_readvariableop_2_resource: <
-quant_dense_1_biasadd_readvariableop_resource:	Y
Oquant_dense_1_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource: [
Qquant_dense_1_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: _
Lquant_dense_2_lastvaluequant_fakequantwithminmaxvars_readvariableop_resource:	X
Nquant_dense_2_lastvaluequant_fakequantwithminmaxvars_readvariableop_1_resource: X
Nquant_dense_2_lastvaluequant_fakequantwithminmaxvars_readvariableop_2_resource: ;
-quant_dense_2_biasadd_readvariableop_resource:Y
Oquant_dense_2_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource: [
Qquant_dense_2_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: 
identity¢/conv2d/kernel/Regularizer/L2Loss/ReadVariableOp¢1conv2d_1/kernel/Regularizer/L2Loss/ReadVariableOp¢1conv2d_2/kernel/Regularizer/L2Loss/ReadVariableOp¢1conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOp¢1conv2d_4/kernel/Regularizer/L2Loss/ReadVariableOp¢.dense/kernel/Regularizer/L2Loss/ReadVariableOp¢0dense_1/kernel/Regularizer/L2Loss/ReadVariableOp¢#quant_conv2d/BiasAdd/ReadVariableOp¢Lquant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp¢Nquant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1¢Nquant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2¢Equant_conv2d/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp¢Gquant_conv2d/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1¢%quant_conv2d_1/BiasAdd/ReadVariableOp¢Nquant_conv2d_1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp¢Pquant_conv2d_1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1¢Pquant_conv2d_1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2¢Gquant_conv2d_1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp¢Iquant_conv2d_1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1¢%quant_conv2d_2/BiasAdd/ReadVariableOp¢Nquant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp¢Pquant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1¢Pquant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2¢Gquant_conv2d_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp¢Iquant_conv2d_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1¢%quant_conv2d_3/BiasAdd/ReadVariableOp¢Nquant_conv2d_3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp¢Pquant_conv2d_3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1¢Pquant_conv2d_3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2¢Gquant_conv2d_3/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp¢Iquant_conv2d_3/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1¢%quant_conv2d_4/BiasAdd/ReadVariableOp¢Nquant_conv2d_4/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp¢Pquant_conv2d_4/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1¢Pquant_conv2d_4/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2¢Gquant_conv2d_4/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp¢Iquant_conv2d_4/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1¢"quant_dense/BiasAdd/ReadVariableOp¢Aquant_dense/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp¢Cquant_dense/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1¢Cquant_dense/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2¢Dquant_dense/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp¢Fquant_dense/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1¢$quant_dense_1/BiasAdd/ReadVariableOp¢Cquant_dense_1/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp¢Equant_dense_1/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1¢Equant_dense_1/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2¢Fquant_dense_1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp¢Hquant_dense_1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1¢$quant_dense_2/BiasAdd/ReadVariableOp¢Cquant_dense_2/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp¢Equant_dense_2/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1¢Equant_dense_2/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2¢Fquant_dense_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp¢Hquant_dense_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1¢Gquantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp¢Iquantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Ð
Gquantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpPquantize_layer_allvaluesquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype0Ô
Iquantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpRquantize_layer_allvaluesquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0²
8quantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsinputsOquantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Qquantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿææê
Lquant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpReadVariableOpUquant_conv2d_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource*&
_output_shapes
: *
dtype0â
Nquant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ReadVariableOpWquant_conv2d_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource*
_output_shapes
: *
dtype0â
Nquant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ReadVariableOpWquant_conv2d_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource*
_output_shapes
: *
dtype0¤
=quant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel!FakeQuantWithMinMaxVarsPerChannelTquant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp:value:0Vquant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1:value:0Vquant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2:value:0*&
_output_shapes
: *
narrow_range(
quant_conv2d/Conv2DConv2DBquantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars:outputs:0Gquant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel:outputs:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿââ *
paddingVALID*
strides

#quant_conv2d/BiasAdd/ReadVariableOpReadVariableOp,quant_conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0¦
quant_conv2d/BiasAddBiasAddquant_conv2d/Conv2D:output:0+quant_conv2d/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿââ t
quant_conv2d/ReluReluquant_conv2d/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿââ Ì
Equant_conv2d/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpNquant_conv2d_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype0Ð
Gquant_conv2d/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpPquant_conv2d_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0Å
6quant_conv2d/MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsquant_conv2d/Relu:activations:0Mquant_conv2d/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Oquant_conv2d/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿââ Õ
quant_max_pooling2d/MaxPoolMaxPool@quant_conv2d/MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿKK *
ksize
*
paddingVALID*
strides
î
Nquant_conv2d_1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpReadVariableOpWquant_conv2d_1_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource*&
_output_shapes
:  *
dtype0æ
Pquant_conv2d_1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ReadVariableOpYquant_conv2d_1_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource*
_output_shapes
: *
dtype0æ
Pquant_conv2d_1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ReadVariableOpYquant_conv2d_1_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource*
_output_shapes
: *
dtype0¬
?quant_conv2d_1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel!FakeQuantWithMinMaxVarsPerChannelVquant_conv2d_1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp:value:0Xquant_conv2d_1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1:value:0Xquant_conv2d_1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2:value:0*&
_output_shapes
:  *
narrow_range(ò
quant_conv2d_1/Conv2DConv2D$quant_max_pooling2d/MaxPool:output:0Iquant_conv2d_1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel:outputs:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿKK *
paddingSAME*
strides

%quant_conv2d_1/BiasAdd/ReadVariableOpReadVariableOp.quant_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0ª
quant_conv2d_1/BiasAddBiasAddquant_conv2d_1/Conv2D:output:0-quant_conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿKK v
quant_conv2d_1/ReluReluquant_conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿKK Ð
Gquant_conv2d_1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpPquant_conv2d_1_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype0Ô
Iquant_conv2d_1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpRquant_conv2d_1_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0Ë
8quant_conv2d_1/MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars!quant_conv2d_1/Relu:activations:0Oquant_conv2d_1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Qquant_conv2d_1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿKK Ù
quant_max_pooling2d_1/MaxPoolMaxPoolBquant_conv2d_1/MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
ksize
*
paddingVALID*
strides
î
Nquant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpReadVariableOpWquant_conv2d_2_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource*&
_output_shapes
: @*
dtype0æ
Pquant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ReadVariableOpYquant_conv2d_2_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource*
_output_shapes
:@*
dtype0æ
Pquant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ReadVariableOpYquant_conv2d_2_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource*
_output_shapes
:@*
dtype0¬
?quant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel!FakeQuantWithMinMaxVarsPerChannelVquant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp:value:0Xquant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1:value:0Xquant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2:value:0*&
_output_shapes
: @*
narrow_range(ô
quant_conv2d_2/Conv2DConv2D&quant_max_pooling2d_1/MaxPool:output:0Iquant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel:outputs:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingSAME*
strides

%quant_conv2d_2/BiasAdd/ReadVariableOpReadVariableOp.quant_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0ª
quant_conv2d_2/BiasAddBiasAddquant_conv2d_2/Conv2D:output:0-quant_conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@v
quant_conv2d_2/ReluReluquant_conv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ð
Gquant_conv2d_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpPquant_conv2d_2_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype0Ô
Iquant_conv2d_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpRquant_conv2d_2_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0Ë
8quant_conv2d_2/MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars!quant_conv2d_2/Relu:activations:0Oquant_conv2d_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Qquant_conv2d_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@î
Nquant_conv2d_3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpReadVariableOpWquant_conv2d_3_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource*&
_output_shapes
:@@*
dtype0æ
Pquant_conv2d_3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ReadVariableOpYquant_conv2d_3_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource*
_output_shapes
:@*
dtype0æ
Pquant_conv2d_3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ReadVariableOpYquant_conv2d_3_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource*
_output_shapes
:@*
dtype0¬
?quant_conv2d_3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel!FakeQuantWithMinMaxVarsPerChannelVquant_conv2d_3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp:value:0Xquant_conv2d_3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1:value:0Xquant_conv2d_3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2:value:0*&
_output_shapes
:@@*
narrow_range(
quant_conv2d_3/Conv2DConv2DBquant_conv2d_2/MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0Iquant_conv2d_3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel:outputs:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingSAME*
strides

%quant_conv2d_3/BiasAdd/ReadVariableOpReadVariableOp.quant_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0ª
quant_conv2d_3/BiasAddBiasAddquant_conv2d_3/Conv2D:output:0-quant_conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@v
quant_conv2d_3/ReluReluquant_conv2d_3/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ð
Gquant_conv2d_3/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpPquant_conv2d_3_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype0Ô
Iquant_conv2d_3/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpRquant_conv2d_3_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0Ë
8quant_conv2d_3/MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars!quant_conv2d_3/Relu:activations:0Oquant_conv2d_3/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Qquant_conv2d_3/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@î
Nquant_conv2d_4/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpReadVariableOpWquant_conv2d_4_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource*&
_output_shapes
:@ *
dtype0æ
Pquant_conv2d_4/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ReadVariableOpYquant_conv2d_4_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource*
_output_shapes
: *
dtype0æ
Pquant_conv2d_4/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ReadVariableOpYquant_conv2d_4_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource*
_output_shapes
: *
dtype0¬
?quant_conv2d_4/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel!FakeQuantWithMinMaxVarsPerChannelVquant_conv2d_4/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp:value:0Xquant_conv2d_4/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1:value:0Xquant_conv2d_4/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2:value:0*&
_output_shapes
:@ *
narrow_range(
quant_conv2d_4/Conv2DConv2DBquant_conv2d_3/MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0Iquant_conv2d_4/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel:outputs:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides

%quant_conv2d_4/BiasAdd/ReadVariableOpReadVariableOp.quant_conv2d_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0ª
quant_conv2d_4/BiasAddBiasAddquant_conv2d_4/Conv2D:output:0-quant_conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ v
quant_conv2d_4/ReluReluquant_conv2d_4/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Ð
Gquant_conv2d_4/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpPquant_conv2d_4_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype0Ô
Iquant_conv2d_4/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpRquant_conv2d_4_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0Ë
8quant_conv2d_4/MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars!quant_conv2d_4/Relu:activations:0Oquant_conv2d_4/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Qquant_conv2d_4/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Ù
quant_max_pooling2d_2/MaxPoolMaxPoolBquant_conv2d_4/MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
ksize
*
paddingVALID*
strides
d
quant_flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   
quant_flatten/ReshapeReshape&quant_max_pooling2d_2/MaxPool:output:0quant_flatten/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÎ
Aquant_dense/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpJquant_dense_lastvaluequant_fakequantwithminmaxvars_readvariableop_resource* 
_output_shapes
:
*
dtype0È
Cquant_dense/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpLquant_dense_lastvaluequant_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0È
Cquant_dense/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2ReadVariableOpLquant_dense_lastvaluequant_fakequantwithminmaxvars_readvariableop_2_resource*
_output_shapes
: *
dtype0è
2quant_dense/LastValueQuant/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsIquant_dense/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Kquant_dense/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0Kquant_dense/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2:value:0* 
_output_shapes
:
*
narrow_range(­
quant_dense/MatMulMatMulquant_flatten/Reshape:output:0<quant_dense/LastValueQuant/FakeQuantWithMinMaxVars:outputs:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"quant_dense/BiasAdd/ReadVariableOpReadVariableOp+quant_dense_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
quant_dense/BiasAddBiasAddquant_dense/MatMul:product:0*quant_dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
quant_dense/ReluReluquant_dense/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÊ
Dquant_dense/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpMquant_dense_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype0Î
Fquant_dense/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpOquant_dense_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0¸
5quant_dense/MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsquant_dense/Relu:activations:0Lquant_dense/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Nquant_dense/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
quant_dropout/IdentityIdentity?quant_dense/MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÒ
Cquant_dense_1/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpLquant_dense_1_lastvaluequant_fakequantwithminmaxvars_readvariableop_resource* 
_output_shapes
:
*
dtype0Ì
Equant_dense_1/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpNquant_dense_1_lastvaluequant_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0Ì
Equant_dense_1/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2ReadVariableOpNquant_dense_1_lastvaluequant_fakequantwithminmaxvars_readvariableop_2_resource*
_output_shapes
: *
dtype0ð
4quant_dense_1/LastValueQuant/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsKquant_dense_1/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Mquant_dense_1/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0Mquant_dense_1/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2:value:0* 
_output_shapes
:
*
narrow_range(²
quant_dense_1/MatMulMatMulquant_dropout/Identity:output:0>quant_dense_1/LastValueQuant/FakeQuantWithMinMaxVars:outputs:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$quant_dense_1/BiasAdd/ReadVariableOpReadVariableOp-quant_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0¡
quant_dense_1/BiasAddBiasAddquant_dense_1/MatMul:product:0,quant_dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿm
quant_dense_1/ReluReluquant_dense_1/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÎ
Fquant_dense_1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpOquant_dense_1_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype0Ò
Hquant_dense_1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpQquant_dense_1_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0À
7quant_dense_1/MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars quant_dense_1/Relu:activations:0Nquant_dense_1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Pquant_dense_1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
quant_dropout_1/IdentityIdentityAquant_dense_1/MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÑ
Cquant_dense_2/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpLquant_dense_2_lastvaluequant_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
:	*
dtype0Ì
Equant_dense_2/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpNquant_dense_2_lastvaluequant_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0Ì
Equant_dense_2/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2ReadVariableOpNquant_dense_2_lastvaluequant_fakequantwithminmaxvars_readvariableop_2_resource*
_output_shapes
: *
dtype0ï
4quant_dense_2/LastValueQuant/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsKquant_dense_2/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Mquant_dense_2/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0Mquant_dense_2/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2:value:0*
_output_shapes
:	*
narrow_range(³
quant_dense_2/MatMulMatMul!quant_dropout_1/Identity:output:0>quant_dense_2/LastValueQuant/FakeQuantWithMinMaxVars:outputs:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$quant_dense_2/BiasAdd/ReadVariableOpReadVariableOp-quant_dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0 
quant_dense_2/BiasAddBiasAddquant_dense_2/MatMul:product:0,quant_dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÎ
Fquant_dense_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpOquant_dense_2_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype0Ò
Hquant_dense_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpQquant_dense_2_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0½
7quant_dense_2/MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsquant_dense_2/BiasAdd:output:0Nquant_dense_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Pquant_dense_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
quant_dense_2/SoftmaxSoftmaxAquant_dense_2/MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÍ
/conv2d/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpUquant_conv2d_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource*&
_output_shapes
: *
dtype0
 conv2d/kernel/Regularizer/L2LossL2Loss7conv2d/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: d
conv2d/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;
conv2d/kernel/Regularizer/mulMul(conv2d/kernel/Regularizer/mul/x:output:0)conv2d/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: Ñ
1conv2d_1/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpWquant_conv2d_1_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource*&
_output_shapes
:  *
dtype0
"conv2d_1/kernel/Regularizer/L2LossL2Loss9conv2d_1/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!conv2d_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o; 
conv2d_1/kernel/Regularizer/mulMul*conv2d_1/kernel/Regularizer/mul/x:output:0+conv2d_1/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: Ñ
1conv2d_2/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpWquant_conv2d_2_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource*&
_output_shapes
: @*
dtype0
"conv2d_2/kernel/Regularizer/L2LossL2Loss9conv2d_2/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!conv2d_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o; 
conv2d_2/kernel/Regularizer/mulMul*conv2d_2/kernel/Regularizer/mul/x:output:0+conv2d_2/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: Ñ
1conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpWquant_conv2d_3_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource*&
_output_shapes
:@@*
dtype0
"conv2d_3/kernel/Regularizer/L2LossL2Loss9conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!conv2d_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o; 
conv2d_3/kernel/Regularizer/mulMul*conv2d_3/kernel/Regularizer/mul/x:output:0+conv2d_3/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: Ñ
1conv2d_4/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpWquant_conv2d_4_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource*&
_output_shapes
:@ *
dtype0
"conv2d_4/kernel/Regularizer/L2LossL2Loss9conv2d_4/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!conv2d_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o; 
conv2d_4/kernel/Regularizer/mulMul*conv2d_4/kernel/Regularizer/mul/x:output:0+conv2d_4/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: »
.dense/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpJquant_dense_lastvaluequant_fakequantwithminmaxvars_readvariableop_resource* 
_output_shapes
:
*
dtype0
dense/kernel/Regularizer/L2LossL2Loss6dense/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: c
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0(dense/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: ¿
0dense_1/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpLquant_dense_1_lastvaluequant_fakequantwithminmaxvars_readvariableop_resource* 
_output_shapes
:
*
dtype0
!dense_1/kernel/Regularizer/L2LossL2Loss8dense_1/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: e
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0*dense_1/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: n
IdentityIdentityquant_dense_2/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ
NoOpNoOp0^conv2d/kernel/Regularizer/L2Loss/ReadVariableOp2^conv2d_1/kernel/Regularizer/L2Loss/ReadVariableOp2^conv2d_2/kernel/Regularizer/L2Loss/ReadVariableOp2^conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOp2^conv2d_4/kernel/Regularizer/L2Loss/ReadVariableOp/^dense/kernel/Regularizer/L2Loss/ReadVariableOp1^dense_1/kernel/Regularizer/L2Loss/ReadVariableOp$^quant_conv2d/BiasAdd/ReadVariableOpM^quant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpO^quant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1O^quant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2F^quant_conv2d/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpH^quant_conv2d/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1&^quant_conv2d_1/BiasAdd/ReadVariableOpO^quant_conv2d_1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpQ^quant_conv2d_1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1Q^quant_conv2d_1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2H^quant_conv2d_1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpJ^quant_conv2d_1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1&^quant_conv2d_2/BiasAdd/ReadVariableOpO^quant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpQ^quant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1Q^quant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2H^quant_conv2d_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpJ^quant_conv2d_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1&^quant_conv2d_3/BiasAdd/ReadVariableOpO^quant_conv2d_3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpQ^quant_conv2d_3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1Q^quant_conv2d_3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2H^quant_conv2d_3/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpJ^quant_conv2d_3/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1&^quant_conv2d_4/BiasAdd/ReadVariableOpO^quant_conv2d_4/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpQ^quant_conv2d_4/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1Q^quant_conv2d_4/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2H^quant_conv2d_4/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpJ^quant_conv2d_4/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1#^quant_dense/BiasAdd/ReadVariableOpB^quant_dense/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpD^quant_dense/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1D^quant_dense/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2E^quant_dense/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpG^quant_dense/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1%^quant_dense_1/BiasAdd/ReadVariableOpD^quant_dense_1/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpF^quant_dense_1/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1F^quant_dense_1/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2G^quant_dense_1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpI^quant_dense_1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1%^quant_dense_2/BiasAdd/ReadVariableOpD^quant_dense_2/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpF^quant_dense_2/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1F^quant_dense_2/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2G^quant_dense_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpI^quant_dense_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1H^quantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOpJ^quantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:ÿÿÿÿÿÿÿÿÿææ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2b
/conv2d/kernel/Regularizer/L2Loss/ReadVariableOp/conv2d/kernel/Regularizer/L2Loss/ReadVariableOp2f
1conv2d_1/kernel/Regularizer/L2Loss/ReadVariableOp1conv2d_1/kernel/Regularizer/L2Loss/ReadVariableOp2f
1conv2d_2/kernel/Regularizer/L2Loss/ReadVariableOp1conv2d_2/kernel/Regularizer/L2Loss/ReadVariableOp2f
1conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOp1conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOp2f
1conv2d_4/kernel/Regularizer/L2Loss/ReadVariableOp1conv2d_4/kernel/Regularizer/L2Loss/ReadVariableOp2`
.dense/kernel/Regularizer/L2Loss/ReadVariableOp.dense/kernel/Regularizer/L2Loss/ReadVariableOp2d
0dense_1/kernel/Regularizer/L2Loss/ReadVariableOp0dense_1/kernel/Regularizer/L2Loss/ReadVariableOp2J
#quant_conv2d/BiasAdd/ReadVariableOp#quant_conv2d/BiasAdd/ReadVariableOp2
Lquant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpLquant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp2 
Nquant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1Nquant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_12 
Nquant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2Nquant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_22
Equant_conv2d/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpEquant_conv2d/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2
Gquant_conv2d/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Gquant_conv2d/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12N
%quant_conv2d_1/BiasAdd/ReadVariableOp%quant_conv2d_1/BiasAdd/ReadVariableOp2 
Nquant_conv2d_1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpNquant_conv2d_1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp2¤
Pquant_conv2d_1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1Pquant_conv2d_1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_12¤
Pquant_conv2d_1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2Pquant_conv2d_1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_22
Gquant_conv2d_1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpGquant_conv2d_1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2
Iquant_conv2d_1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Iquant_conv2d_1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12N
%quant_conv2d_2/BiasAdd/ReadVariableOp%quant_conv2d_2/BiasAdd/ReadVariableOp2 
Nquant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpNquant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp2¤
Pquant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1Pquant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_12¤
Pquant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2Pquant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_22
Gquant_conv2d_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpGquant_conv2d_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2
Iquant_conv2d_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Iquant_conv2d_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12N
%quant_conv2d_3/BiasAdd/ReadVariableOp%quant_conv2d_3/BiasAdd/ReadVariableOp2 
Nquant_conv2d_3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpNquant_conv2d_3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp2¤
Pquant_conv2d_3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1Pquant_conv2d_3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_12¤
Pquant_conv2d_3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2Pquant_conv2d_3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_22
Gquant_conv2d_3/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpGquant_conv2d_3/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2
Iquant_conv2d_3/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Iquant_conv2d_3/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12N
%quant_conv2d_4/BiasAdd/ReadVariableOp%quant_conv2d_4/BiasAdd/ReadVariableOp2 
Nquant_conv2d_4/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpNquant_conv2d_4/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp2¤
Pquant_conv2d_4/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1Pquant_conv2d_4/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_12¤
Pquant_conv2d_4/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2Pquant_conv2d_4/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_22
Gquant_conv2d_4/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpGquant_conv2d_4/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2
Iquant_conv2d_4/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Iquant_conv2d_4/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12H
"quant_dense/BiasAdd/ReadVariableOp"quant_dense/BiasAdd/ReadVariableOp2
Aquant_dense/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpAquant_dense/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp2
Cquant_dense/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1Cquant_dense/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_12
Cquant_dense/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2Cquant_dense/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_22
Dquant_dense/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpDquant_dense/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2
Fquant_dense/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Fquant_dense/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12L
$quant_dense_1/BiasAdd/ReadVariableOp$quant_dense_1/BiasAdd/ReadVariableOp2
Cquant_dense_1/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpCquant_dense_1/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp2
Equant_dense_1/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1Equant_dense_1/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_12
Equant_dense_1/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2Equant_dense_1/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_22
Fquant_dense_1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpFquant_dense_1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2
Hquant_dense_1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Hquant_dense_1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12L
$quant_dense_2/BiasAdd/ReadVariableOp$quant_dense_2/BiasAdd/ReadVariableOp2
Cquant_dense_2/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpCquant_dense_2/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp2
Equant_dense_2/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1Equant_dense_2/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_12
Equant_dense_2/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2Equant_dense_2/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_22
Fquant_dense_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpFquant_dense_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2
Hquant_dense_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Hquant_dense_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12
Gquantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOpGquantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2
Iquantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Iquantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿææ
 
_user_specified_nameinputs
	

/__inference_quant_conv2d_1_layer_call_fn_146236

inputs!
unknown:  
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: 
identity¢StatefulPartitionedCall©
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿKK *(
_read_only_resource_inputs

*;
config_proto+)

CPU

GPU2*0J
	
   E 8 *S
fNRL
J__inference_quant_conv2d_1_layer_call_and_return_conditional_losses_143027w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿKK `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿKK : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿKK 
 
_user_specified_nameinputs
å
P
4__inference_quant_max_pooling2d_layer_call_fn_146209

inputs
identityÐ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿKK * 
_read_only_resource_inputs
 *;
config_proto+)

CPU

GPU2*0J
	
   E 8 *X
fSRQ
O__inference_quant_max_pooling2d_layer_call_and_return_conditional_losses_144159h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿKK "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿââ :Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿââ 
 
_user_specified_nameinputs
Ú

F__inference_sequential_layer_call_and_return_conditional_losses_144544

inputs
quantize_layer_144400: 
quantize_layer_144402: -
quant_conv2d_144405: !
quant_conv2d_144407: !
quant_conv2d_144409: !
quant_conv2d_144411: 
quant_conv2d_144413: 
quant_conv2d_144415: /
quant_conv2d_1_144419:  #
quant_conv2d_1_144421: #
quant_conv2d_1_144423: #
quant_conv2d_1_144425: 
quant_conv2d_1_144427: 
quant_conv2d_1_144429: /
quant_conv2d_2_144433: @#
quant_conv2d_2_144435:@#
quant_conv2d_2_144437:@#
quant_conv2d_2_144439:@
quant_conv2d_2_144441: 
quant_conv2d_2_144443: /
quant_conv2d_3_144446:@@#
quant_conv2d_3_144448:@#
quant_conv2d_3_144450:@#
quant_conv2d_3_144452:@
quant_conv2d_3_144454: 
quant_conv2d_3_144456: /
quant_conv2d_4_144459:@ #
quant_conv2d_4_144461: #
quant_conv2d_4_144463: #
quant_conv2d_4_144465: 
quant_conv2d_4_144467: 
quant_conv2d_4_144469: &
quant_dense_144474:

quant_dense_144476: 
quant_dense_144478: !
quant_dense_144480:	
quant_dense_144482: 
quant_dense_144484: (
quant_dense_1_144488:

quant_dense_1_144490: 
quant_dense_1_144492: #
quant_dense_1_144494:	
quant_dense_1_144496: 
quant_dense_1_144498: '
quant_dense_2_144502:	
quant_dense_2_144504: 
quant_dense_2_144506: "
quant_dense_2_144508:
quant_dense_2_144510: 
quant_dense_2_144512: 
identity¢/conv2d/kernel/Regularizer/L2Loss/ReadVariableOp¢1conv2d_1/kernel/Regularizer/L2Loss/ReadVariableOp¢1conv2d_2/kernel/Regularizer/L2Loss/ReadVariableOp¢1conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOp¢1conv2d_4/kernel/Regularizer/L2Loss/ReadVariableOp¢.dense/kernel/Regularizer/L2Loss/ReadVariableOp¢0dense_1/kernel/Regularizer/L2Loss/ReadVariableOp¢$quant_conv2d/StatefulPartitionedCall¢&quant_conv2d_1/StatefulPartitionedCall¢&quant_conv2d_2/StatefulPartitionedCall¢&quant_conv2d_3/StatefulPartitionedCall¢&quant_conv2d_4/StatefulPartitionedCall¢#quant_dense/StatefulPartitionedCall¢%quant_dense_1/StatefulPartitionedCall¢%quant_dense_2/StatefulPartitionedCall¢%quant_dropout/StatefulPartitionedCall¢'quant_dropout_1/StatefulPartitionedCall¢&quantize_layer/StatefulPartitionedCall
&quantize_layer/StatefulPartitionedCallStatefulPartitionedCallinputsquantize_layer_144400quantize_layer_144402*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿææ* 
_read_only_resource_inputs
 *;
config_proto+)

CPU

GPU2*0J
	
   E 8 *S
fNRL
J__inference_quantize_layer_layer_call_and_return_conditional_losses_144283
$quant_conv2d/StatefulPartitionedCallStatefulPartitionedCall/quantize_layer/StatefulPartitionedCall:output:0quant_conv2d_144405quant_conv2d_144407quant_conv2d_144409quant_conv2d_144411quant_conv2d_144413quant_conv2d_144415*
Tin
	2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿââ *$
_read_only_resource_inputs
*;
config_proto+)

CPU

GPU2*0J
	
   E 8 *Q
fLRJ
H__inference_quant_conv2d_layer_call_and_return_conditional_losses_144235
#quant_max_pooling2d/PartitionedCallPartitionedCall-quant_conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿKK * 
_read_only_resource_inputs
 *;
config_proto+)

CPU

GPU2*0J
	
   E 8 *X
fSRQ
O__inference_quant_max_pooling2d_layer_call_and_return_conditional_losses_144159¤
&quant_conv2d_1/StatefulPartitionedCallStatefulPartitionedCall,quant_max_pooling2d/PartitionedCall:output:0quant_conv2d_1_144419quant_conv2d_1_144421quant_conv2d_1_144423quant_conv2d_1_144425quant_conv2d_1_144427quant_conv2d_1_144429*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿKK *$
_read_only_resource_inputs
*;
config_proto+)

CPU

GPU2*0J
	
   E 8 *S
fNRL
J__inference_quant_conv2d_1_layer_call_and_return_conditional_losses_144131
%quant_max_pooling2d_1/PartitionedCallPartitionedCall/quant_conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *;
config_proto+)

CPU

GPU2*0J
	
   E 8 *Z
fURS
Q__inference_quant_max_pooling2d_1_layer_call_and_return_conditional_losses_144055¦
&quant_conv2d_2/StatefulPartitionedCallStatefulPartitionedCall.quant_max_pooling2d_1/PartitionedCall:output:0quant_conv2d_2_144433quant_conv2d_2_144435quant_conv2d_2_144437quant_conv2d_2_144439quant_conv2d_2_144441quant_conv2d_2_144443*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*;
config_proto+)

CPU

GPU2*0J
	
   E 8 *S
fNRL
J__inference_quant_conv2d_2_layer_call_and_return_conditional_losses_144027§
&quant_conv2d_3/StatefulPartitionedCallStatefulPartitionedCall/quant_conv2d_2/StatefulPartitionedCall:output:0quant_conv2d_3_144446quant_conv2d_3_144448quant_conv2d_3_144450quant_conv2d_3_144452quant_conv2d_3_144454quant_conv2d_3_144456*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*;
config_proto+)

CPU

GPU2*0J
	
   E 8 *S
fNRL
J__inference_quant_conv2d_3_layer_call_and_return_conditional_losses_143939§
&quant_conv2d_4/StatefulPartitionedCallStatefulPartitionedCall/quant_conv2d_3/StatefulPartitionedCall:output:0quant_conv2d_4_144459quant_conv2d_4_144461quant_conv2d_4_144463quant_conv2d_4_144465quant_conv2d_4_144467quant_conv2d_4_144469*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*;
config_proto+)

CPU

GPU2*0J
	
   E 8 *S
fNRL
J__inference_quant_conv2d_4_layer_call_and_return_conditional_losses_143851
%quant_max_pooling2d_2/PartitionedCallPartitionedCall/quant_conv2d_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *;
config_proto+)

CPU

GPU2*0J
	
   E 8 *Z
fURS
Q__inference_quant_max_pooling2d_2_layer_call_and_return_conditional_losses_143775ù
quant_flatten/PartitionedCallPartitionedCall.quant_max_pooling2d_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *;
config_proto+)

CPU

GPU2*0J
	
   E 8 *R
fMRK
I__inference_quant_flatten_layer_call_and_return_conditional_losses_143759ÿ
#quant_dense/StatefulPartitionedCallStatefulPartitionedCall&quant_flatten/PartitionedCall:output:0quant_dense_144474quant_dense_144476quant_dense_144478quant_dense_144480quant_dense_144482quant_dense_144484*
Tin
	2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*;
config_proto+)

CPU

GPU2*0J
	
   E 8 *P
fKRI
G__inference_quant_dense_layer_call_and_return_conditional_losses_143730
%quant_dropout/StatefulPartitionedCallStatefulPartitionedCall,quant_dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *;
config_proto+)

CPU

GPU2*0J
	
   E 8 *R
fMRK
I__inference_quant_dropout_layer_call_and_return_conditional_losses_143654
%quant_dense_1/StatefulPartitionedCallStatefulPartitionedCall.quant_dropout/StatefulPartitionedCall:output:0quant_dense_1_144488quant_dense_1_144490quant_dense_1_144492quant_dense_1_144494quant_dense_1_144496quant_dense_1_144498*
Tin
	2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*;
config_proto+)

CPU

GPU2*0J
	
   E 8 *R
fMRK
I__inference_quant_dense_1_layer_call_and_return_conditional_losses_143619µ
'quant_dropout_1/StatefulPartitionedCallStatefulPartitionedCall.quant_dense_1/StatefulPartitionedCall:output:0&^quant_dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *;
config_proto+)

CPU

GPU2*0J
	
   E 8 *T
fORM
K__inference_quant_dropout_1_layer_call_and_return_conditional_losses_143543
%quant_dense_2/StatefulPartitionedCallStatefulPartitionedCall0quant_dropout_1/StatefulPartitionedCall:output:0quant_dense_2_144502quant_dense_2_144504quant_dense_2_144506quant_dense_2_144508quant_dense_2_144510quant_dense_2_144512*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*;
config_proto+)

CPU

GPU2*0J
	
   E 8 *R
fMRK
I__inference_quant_dense_2_layer_call_and_return_conditional_losses_143508
/conv2d/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpquant_conv2d_144405*&
_output_shapes
: *
dtype0
 conv2d/kernel/Regularizer/L2LossL2Loss7conv2d/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: d
conv2d/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;
conv2d/kernel/Regularizer/mulMul(conv2d/kernel/Regularizer/mul/x:output:0)conv2d/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: 
1conv2d_1/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpquant_conv2d_1_144419*&
_output_shapes
:  *
dtype0
"conv2d_1/kernel/Regularizer/L2LossL2Loss9conv2d_1/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!conv2d_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o; 
conv2d_1/kernel/Regularizer/mulMul*conv2d_1/kernel/Regularizer/mul/x:output:0+conv2d_1/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: 
1conv2d_2/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpquant_conv2d_2_144433*&
_output_shapes
: @*
dtype0
"conv2d_2/kernel/Regularizer/L2LossL2Loss9conv2d_2/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!conv2d_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o; 
conv2d_2/kernel/Regularizer/mulMul*conv2d_2/kernel/Regularizer/mul/x:output:0+conv2d_2/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: 
1conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpquant_conv2d_3_144446*&
_output_shapes
:@@*
dtype0
"conv2d_3/kernel/Regularizer/L2LossL2Loss9conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!conv2d_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o; 
conv2d_3/kernel/Regularizer/mulMul*conv2d_3/kernel/Regularizer/mul/x:output:0+conv2d_3/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: 
1conv2d_4/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpquant_conv2d_4_144459*&
_output_shapes
:@ *
dtype0
"conv2d_4/kernel/Regularizer/L2LossL2Loss9conv2d_4/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!conv2d_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o; 
conv2d_4/kernel/Regularizer/mulMul*conv2d_4/kernel/Regularizer/mul/x:output:0+conv2d_4/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: 
.dense/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpquant_dense_144474* 
_output_shapes
:
*
dtype0
dense/kernel/Regularizer/L2LossL2Loss6dense/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: c
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0(dense/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: 
0dense_1/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpquant_dense_1_144488* 
_output_shapes
:
*
dtype0
!dense_1/kernel/Regularizer/L2LossL2Loss8dense_1/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: e
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0*dense_1/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: }
IdentityIdentity.quant_dense_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿè
NoOpNoOp0^conv2d/kernel/Regularizer/L2Loss/ReadVariableOp2^conv2d_1/kernel/Regularizer/L2Loss/ReadVariableOp2^conv2d_2/kernel/Regularizer/L2Loss/ReadVariableOp2^conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOp2^conv2d_4/kernel/Regularizer/L2Loss/ReadVariableOp/^dense/kernel/Regularizer/L2Loss/ReadVariableOp1^dense_1/kernel/Regularizer/L2Loss/ReadVariableOp%^quant_conv2d/StatefulPartitionedCall'^quant_conv2d_1/StatefulPartitionedCall'^quant_conv2d_2/StatefulPartitionedCall'^quant_conv2d_3/StatefulPartitionedCall'^quant_conv2d_4/StatefulPartitionedCall$^quant_dense/StatefulPartitionedCall&^quant_dense_1/StatefulPartitionedCall&^quant_dense_2/StatefulPartitionedCall&^quant_dropout/StatefulPartitionedCall(^quant_dropout_1/StatefulPartitionedCall'^quantize_layer/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:ÿÿÿÿÿÿÿÿÿææ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2b
/conv2d/kernel/Regularizer/L2Loss/ReadVariableOp/conv2d/kernel/Regularizer/L2Loss/ReadVariableOp2f
1conv2d_1/kernel/Regularizer/L2Loss/ReadVariableOp1conv2d_1/kernel/Regularizer/L2Loss/ReadVariableOp2f
1conv2d_2/kernel/Regularizer/L2Loss/ReadVariableOp1conv2d_2/kernel/Regularizer/L2Loss/ReadVariableOp2f
1conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOp1conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOp2f
1conv2d_4/kernel/Regularizer/L2Loss/ReadVariableOp1conv2d_4/kernel/Regularizer/L2Loss/ReadVariableOp2`
.dense/kernel/Regularizer/L2Loss/ReadVariableOp.dense/kernel/Regularizer/L2Loss/ReadVariableOp2d
0dense_1/kernel/Regularizer/L2Loss/ReadVariableOp0dense_1/kernel/Regularizer/L2Loss/ReadVariableOp2L
$quant_conv2d/StatefulPartitionedCall$quant_conv2d/StatefulPartitionedCall2P
&quant_conv2d_1/StatefulPartitionedCall&quant_conv2d_1/StatefulPartitionedCall2P
&quant_conv2d_2/StatefulPartitionedCall&quant_conv2d_2/StatefulPartitionedCall2P
&quant_conv2d_3/StatefulPartitionedCall&quant_conv2d_3/StatefulPartitionedCall2P
&quant_conv2d_4/StatefulPartitionedCall&quant_conv2d_4/StatefulPartitionedCall2J
#quant_dense/StatefulPartitionedCall#quant_dense/StatefulPartitionedCall2N
%quant_dense_1/StatefulPartitionedCall%quant_dense_1/StatefulPartitionedCall2N
%quant_dense_2/StatefulPartitionedCall%quant_dense_2/StatefulPartitionedCall2N
%quant_dropout/StatefulPartitionedCall%quant_dropout/StatefulPartitionedCall2R
'quant_dropout_1/StatefulPartitionedCall'quant_dropout_1/StatefulPartitionedCall2P
&quantize_layer/StatefulPartitionedCall&quantize_layer/StatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿææ
 
_user_specified_nameinputs
Ë
e
I__inference_quant_flatten_layer_call_and_return_conditional_losses_143759

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ :W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Ã 
Û
I__inference_quant_dense_2_layer_call_and_return_conditional_losses_143293

inputsQ
>lastvaluequant_fakequantwithminmaxvars_readvariableop_resource:	J
@lastvaluequant_fakequantwithminmaxvars_readvariableop_1_resource: J
@lastvaluequant_fakequantwithminmaxvars_readvariableop_2_resource: -
biasadd_readvariableop_resource:K
Amovingavgquantize_fakequantwithminmaxvars_readvariableop_resource: M
Cmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: 
identity¢BiasAdd/ReadVariableOp¢5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp¢7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1¢7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2¢8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp¢:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1µ
5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp>lastvaluequant_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
:	*
dtype0°
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp@lastvaluequant_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0°
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2ReadVariableOp@lastvaluequant_fakequantwithminmaxvars_readvariableop_2_resource*
_output_shapes
: *
dtype0·
&LastValueQuant/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars=LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp:value:0?LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0?LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2:value:0*
_output_shapes
:	*
narrow_range(|
MatMulMatMulinputs0LastValueQuant/FakeQuantWithMinMaxVars:outputs:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ²
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpAmovingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype0¶
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpCmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0
)MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsBiasAdd:output:0@MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BMovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿy
SoftmaxSoftmax3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp^BiasAdd/ReadVariableOp6^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp8^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_18^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_29^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ: : : : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2n
5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp2r
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_17LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_12r
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_27LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_22t
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ñ
ò
.__inference_quant_dense_1_layer_call_fn_146905

inputs
unknown:

	unknown_0: 
	unknown_1: 
	unknown_2:	
	unknown_3: 
	unknown_4: 
identity¢StatefulPartitionedCall¡
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*;
config_proto+)

CPU

GPU2*0J
	
   E 8 *R
fMRK
I__inference_quant_dense_1_layer_call_and_return_conditional_losses_143251p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¯
k
O__inference_quant_max_pooling2d_layer_call_and_return_conditional_losses_146219

inputs
identity
MaxPoolMaxPoolinputs*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿKK *
ksize
*
paddingVALID*
strides
`
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿKK "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿââ :Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿââ 
 
_user_specified_nameinputs
à
g
I__inference_quant_dropout_layer_call_and_return_conditional_losses_143224

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
í
ð
.__inference_quant_dense_2_layer_call_fn_147048

inputs
unknown:	
	unknown_0: 
	unknown_1: 
	unknown_2:
	unknown_3: 
	unknown_4: 
identity¢StatefulPartitionedCall 
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*;
config_proto+)

CPU

GPU2*0J
	
   E 8 *R
fMRK
I__inference_quant_dense_2_layer_call_and_return_conditional_losses_143293o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
³N
Ø
I__inference_quant_dense_2_layer_call_and_return_conditional_losses_143508

inputsB
/lastvaluequant_batchmin_readvariableop_resource:	/
%lastvaluequant_assignminlast_resource: /
%lastvaluequant_assignmaxlast_resource: -
biasadd_readvariableop_resource:@
6movingavgquantize_assignminema_readvariableop_resource: @
6movingavgquantize_assignmaxema_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢LastValueQuant/AssignMaxLast¢LastValueQuant/AssignMinLast¢&LastValueQuant/BatchMax/ReadVariableOp¢&LastValueQuant/BatchMin/ReadVariableOp¢5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp¢7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1¢7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2¢2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp¢-MovingAvgQuantize/AssignMaxEma/ReadVariableOp¢2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp¢-MovingAvgQuantize/AssignMinEma/ReadVariableOp¢8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp¢:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1e
LastValueQuant/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
&LastValueQuant/BatchMin/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource*
_output_shapes
:	*
dtype0
LastValueQuant/BatchMinMin.LastValueQuant/BatchMin/ReadVariableOp:value:0LastValueQuant/Const:output:0*
T0*
_output_shapes
: g
LastValueQuant/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       
&LastValueQuant/BatchMax/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource*
_output_shapes
:	*
dtype0
LastValueQuant/BatchMaxMax.LastValueQuant/BatchMax/ReadVariableOp:value:0LastValueQuant/Const_1:output:0*
T0*
_output_shapes
: ]
LastValueQuant/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ¿
LastValueQuant/truedivRealDiv LastValueQuant/BatchMax:output:0!LastValueQuant/truediv/y:output:0*
T0*
_output_shapes
: 
LastValueQuant/MinimumMinimum LastValueQuant/BatchMin:output:0LastValueQuant/truediv:z:0*
T0*
_output_shapes
: Y
LastValueQuant/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ¿{
LastValueQuant/mulMul LastValueQuant/BatchMin:output:0LastValueQuant/mul/y:output:0*
T0*
_output_shapes
: |
LastValueQuant/MaximumMaximum LastValueQuant/BatchMax:output:0LastValueQuant/mul:z:0*
T0*
_output_shapes
: ¬
LastValueQuant/AssignMinLastAssignVariableOp%lastvaluequant_assignminlast_resourceLastValueQuant/Minimum:z:0*
_output_shapes
 *
dtype0*
validate_shape(¬
LastValueQuant/AssignMaxLastAssignVariableOp%lastvaluequant_assignmaxlast_resourceLastValueQuant/Maximum:z:0*
_output_shapes
 *
dtype0*
validate_shape(¦
5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource*
_output_shapes
:	*
dtype0´
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp%lastvaluequant_assignminlast_resource^LastValueQuant/AssignMinLast*
_output_shapes
: *
dtype0´
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2ReadVariableOp%lastvaluequant_assignmaxlast_resource^LastValueQuant/AssignMaxLast*
_output_shapes
: *
dtype0·
&LastValueQuant/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars=LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp:value:0?LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0?LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2:value:0*
_output_shapes
:	*
narrow_range(|
MatMulMatMulinputs0LastValueQuant/FakeQuantWithMinMaxVars:outputs:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
MovingAvgQuantize/ConstConst*
_output_shapes
:*
dtype0*
valueB"       v
MovingAvgQuantize/BatchMinMinBiasAdd:output:0 MovingAvgQuantize/Const:output:0*
T0*
_output_shapes
: j
MovingAvgQuantize/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       x
MovingAvgQuantize/BatchMaxMaxBiasAdd:output:0"MovingAvgQuantize/Const_1:output:0*
T0*
_output_shapes
: `
MovingAvgQuantize/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
MovingAvgQuantize/MinimumMinimum#MovingAvgQuantize/BatchMin:output:0$MovingAvgQuantize/Minimum/y:output:0*
T0*
_output_shapes
: `
MovingAvgQuantize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
MovingAvgQuantize/MaximumMaximum#MovingAvgQuantize/BatchMax:output:0$MovingAvgQuantize/Maximum/y:output:0*
T0*
_output_shapes
: i
$MovingAvgQuantize/AssignMinEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o:
-MovingAvgQuantize/AssignMinEma/ReadVariableOpReadVariableOp6movingavgquantize_assignminema_readvariableop_resource*
_output_shapes
: *
dtype0 
"MovingAvgQuantize/AssignMinEma/subSub5MovingAvgQuantize/AssignMinEma/ReadVariableOp:value:0MovingAvgQuantize/Minimum:z:0*
T0*
_output_shapes
: ¡
"MovingAvgQuantize/AssignMinEma/mulMul&MovingAvgQuantize/AssignMinEma/sub:z:0-MovingAvgQuantize/AssignMinEma/decay:output:0*
T0*
_output_shapes
: ü
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOpAssignSubVariableOp6movingavgquantize_assignminema_readvariableop_resource&MovingAvgQuantize/AssignMinEma/mul:z:0.^MovingAvgQuantize/AssignMinEma/ReadVariableOp*
_output_shapes
 *
dtype0i
$MovingAvgQuantize/AssignMaxEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o:
-MovingAvgQuantize/AssignMaxEma/ReadVariableOpReadVariableOp6movingavgquantize_assignmaxema_readvariableop_resource*
_output_shapes
: *
dtype0 
"MovingAvgQuantize/AssignMaxEma/subSub5MovingAvgQuantize/AssignMaxEma/ReadVariableOp:value:0MovingAvgQuantize/Maximum:z:0*
T0*
_output_shapes
: ¡
"MovingAvgQuantize/AssignMaxEma/mulMul&MovingAvgQuantize/AssignMaxEma/sub:z:0-MovingAvgQuantize/AssignMaxEma/decay:output:0*
T0*
_output_shapes
: ü
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpAssignSubVariableOp6movingavgquantize_assignmaxema_readvariableop_resource&MovingAvgQuantize/AssignMaxEma/mul:z:0.^MovingAvgQuantize/AssignMaxEma/ReadVariableOp*
_output_shapes
 *
dtype0Ü
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp6movingavgquantize_assignminema_readvariableop_resource3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp*
_output_shapes
: *
dtype0Þ
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp6movingavgquantize_assignmaxema_readvariableop_resource3^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp*
_output_shapes
: *
dtype0
)MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsBiasAdd:output:0@MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BMovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿy
SoftmaxSoftmax3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÝ
NoOpNoOp^BiasAdd/ReadVariableOp^LastValueQuant/AssignMaxLast^LastValueQuant/AssignMinLast'^LastValueQuant/BatchMax/ReadVariableOp'^LastValueQuant/BatchMin/ReadVariableOp6^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp8^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_18^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_23^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp.^MovingAvgQuantize/AssignMaxEma/ReadVariableOp3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp.^MovingAvgQuantize/AssignMinEma/ReadVariableOp9^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ: : : : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2<
LastValueQuant/AssignMaxLastLastValueQuant/AssignMaxLast2<
LastValueQuant/AssignMinLastLastValueQuant/AssignMinLast2P
&LastValueQuant/BatchMax/ReadVariableOp&LastValueQuant/BatchMax/ReadVariableOp2P
&LastValueQuant/BatchMin/ReadVariableOp&LastValueQuant/BatchMin/ReadVariableOp2n
5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp2r
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_17LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_12r
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_27LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_22h
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2^
-MovingAvgQuantize/AssignMaxEma/ReadVariableOp-MovingAvgQuantize/AssignMaxEma/ReadVariableOp2h
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2^
-MovingAvgQuantize/AssignMinEma/ReadVariableOp-MovingAvgQuantize/AssignMinEma/ReadVariableOp2t
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
	

/__inference_quant_conv2d_1_layer_call_fn_146253

inputs!
unknown:  
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: 
identity¢StatefulPartitionedCall¥
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿKK *$
_read_only_resource_inputs
*;
config_proto+)

CPU

GPU2*0J
	
   E 8 *S
fNRL
J__inference_quant_conv2d_1_layer_call_and_return_conditional_losses_144131w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿKK `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿKK : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿKK 
 
_user_specified_nameinputs
å
R
6__inference_quant_max_pooling2d_2_layer_call_fn_146713

inputs
identityÒ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *;
config_proto+)

CPU

GPU2*0J
	
   E 8 *Z
fURS
Q__inference_quant_max_pooling2d_2_layer_call_and_return_conditional_losses_143775h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ :W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Æ
L
0__inference_max_pooling2d_1_layer_call_fn_147168

inputs
identityç
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *;
config_proto+)

CPU

GPU2*0J
	
   E 8 *T
fORM
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_142919
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

g
K__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_147210

inputs
identity¢
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¹
J
.__inference_quant_dropout_layer_call_fn_146866

inputs
identityÃ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *;
config_proto+)

CPU

GPU2*0J
	
   E 8 *R
fMRK
I__inference_quant_dropout_layer_call_and_return_conditional_losses_143224a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


j
K__inference_quant_dropout_1_layer_call_and_return_conditional_losses_147031

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?§
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿp
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ÿ)
Û
J__inference_quant_conv2d_4_layer_call_and_return_conditional_losses_143151

inputsb
Hlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource:@ X
Jlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource: X
Jlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource: -
biasadd_readvariableop_resource: K
Amovingavgquantize_fakequantwithminmaxvars_readvariableop_resource: M
Cmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: 
identity¢BiasAdd/ReadVariableOp¢?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp¢ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1¢ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2¢8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp¢:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1¢1conv2d_4/kernel/Regularizer/L2Loss/ReadVariableOpÐ
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpReadVariableOpHlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource*&
_output_shapes
:@ *
dtype0È
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ReadVariableOpJlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource*
_output_shapes
: *
dtype0È
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ReadVariableOpJlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource*
_output_shapes
: *
dtype0ð
0LastValueQuant/FakeQuantWithMinMaxVarsPerChannel!FakeQuantWithMinMaxVarsPerChannelGLastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2:value:0*&
_output_shapes
:@ *
narrow_range(¶
Conv2DConv2Dinputs:LastValueQuant/FakeQuantWithMinMaxVarsPerChannel:outputs:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ²
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpAmovingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype0¶
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpCmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0
)MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsRelu:activations:0@MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BMovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Â
1conv2d_4/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpHlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource*&
_output_shapes
:@ *
dtype0
"conv2d_4/kernel/Regularizer/L2LossL2Loss9conv2d_4/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!conv2d_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o; 
conv2d_4/kernel/Regularizer/mulMul*conv2d_4/kernel/Regularizer/mul/x:output:0+conv2d_4/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: 
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Õ
NoOpNoOp^BiasAdd/ReadVariableOp@^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpB^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1B^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_29^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12^conv2d_4/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ@: : : : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp2
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_12
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_22t
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12f
1conv2d_4/kernel/Regularizer/L2Loss/ReadVariableOp1conv2d_4/kernel/Regularizer/L2Loss/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
ÿ)
Û
J__inference_quant_conv2d_3_layer_call_and_return_conditional_losses_146530

inputsb
Hlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource:@@X
Jlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource:@X
Jlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource:@-
biasadd_readvariableop_resource:@K
Amovingavgquantize_fakequantwithminmaxvars_readvariableop_resource: M
Cmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: 
identity¢BiasAdd/ReadVariableOp¢?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp¢ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1¢ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2¢8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp¢:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1¢1conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOpÐ
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpReadVariableOpHlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource*&
_output_shapes
:@@*
dtype0È
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ReadVariableOpJlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource*
_output_shapes
:@*
dtype0È
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ReadVariableOpJlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource*
_output_shapes
:@*
dtype0ð
0LastValueQuant/FakeQuantWithMinMaxVarsPerChannel!FakeQuantWithMinMaxVarsPerChannelGLastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2:value:0*&
_output_shapes
:@@*
narrow_range(¶
Conv2DConv2Dinputs:LastValueQuant/FakeQuantWithMinMaxVarsPerChannel:outputs:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@²
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpAmovingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype0¶
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpCmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0
)MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsRelu:activations:0@MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BMovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Â
1conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpHlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource*&
_output_shapes
:@@*
dtype0
"conv2d_3/kernel/Regularizer/L2LossL2Loss9conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!conv2d_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o; 
conv2d_3/kernel/Regularizer/mulMul*conv2d_3/kernel/Regularizer/mul/x:output:0+conv2d_3/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: 
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Õ
NoOpNoOp^BiasAdd/ReadVariableOp@^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpB^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1B^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_29^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12^conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ@: : : : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp2
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_12
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_22t
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12f
1conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOp1conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
ÿ)
Û
J__inference_quant_conv2d_2_layer_call_and_return_conditional_losses_143073

inputsb
Hlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource: @X
Jlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource:@X
Jlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource:@-
biasadd_readvariableop_resource:@K
Amovingavgquantize_fakequantwithminmaxvars_readvariableop_resource: M
Cmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: 
identity¢BiasAdd/ReadVariableOp¢?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp¢ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1¢ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2¢8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp¢:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1¢1conv2d_2/kernel/Regularizer/L2Loss/ReadVariableOpÐ
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpReadVariableOpHlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource*&
_output_shapes
: @*
dtype0È
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ReadVariableOpJlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource*
_output_shapes
:@*
dtype0È
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ReadVariableOpJlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource*
_output_shapes
:@*
dtype0ð
0LastValueQuant/FakeQuantWithMinMaxVarsPerChannel!FakeQuantWithMinMaxVarsPerChannelGLastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2:value:0*&
_output_shapes
: @*
narrow_range(¶
Conv2DConv2Dinputs:LastValueQuant/FakeQuantWithMinMaxVarsPerChannel:outputs:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@²
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpAmovingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype0¶
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpCmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0
)MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsRelu:activations:0@MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BMovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Â
1conv2d_2/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpHlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource*&
_output_shapes
: @*
dtype0
"conv2d_2/kernel/Regularizer/L2LossL2Loss9conv2d_2/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!conv2d_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o; 
conv2d_2/kernel/Regularizer/mulMul*conv2d_2/kernel/Regularizer/mul/x:output:0+conv2d_2/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: 
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Õ
NoOpNoOp^BiasAdd/ReadVariableOp@^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpB^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1B^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_29^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12^conv2d_2/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ : : : : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp2
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_12
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_22t
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12f
1conv2d_2/kernel/Regularizer/L2Loss/ReadVariableOp1conv2d_2/kernel/Regularizer/L2Loss/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
#
ý
J__inference_quantize_layer_layer_call_and_return_conditional_losses_144283

inputs;
1allvaluesquantize_minimum_readvariableop_resource: ;
1allvaluesquantize_maximum_readvariableop_resource: 
identity¢#AllValuesQuantize/AssignMaxAllValue¢#AllValuesQuantize/AssignMinAllValue¢8AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp¢:AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1¢(AllValuesQuantize/Maximum/ReadVariableOp¢(AllValuesQuantize/Minimum/ReadVariableOpp
AllValuesQuantize/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             l
AllValuesQuantize/BatchMinMininputs AllValuesQuantize/Const:output:0*
T0*
_output_shapes
: r
AllValuesQuantize/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             n
AllValuesQuantize/BatchMaxMaxinputs"AllValuesQuantize/Const_1:output:0*
T0*
_output_shapes
: 
(AllValuesQuantize/Minimum/ReadVariableOpReadVariableOp1allvaluesquantize_minimum_readvariableop_resource*
_output_shapes
: *
dtype0
AllValuesQuantize/MinimumMinimum0AllValuesQuantize/Minimum/ReadVariableOp:value:0#AllValuesQuantize/BatchMin:output:0*
T0*
_output_shapes
: b
AllValuesQuantize/Minimum_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
AllValuesQuantize/Minimum_1MinimumAllValuesQuantize/Minimum:z:0&AllValuesQuantize/Minimum_1/y:output:0*
T0*
_output_shapes
: 
(AllValuesQuantize/Maximum/ReadVariableOpReadVariableOp1allvaluesquantize_maximum_readvariableop_resource*
_output_shapes
: *
dtype0
AllValuesQuantize/MaximumMaximum0AllValuesQuantize/Maximum/ReadVariableOp:value:0#AllValuesQuantize/BatchMax:output:0*
T0*
_output_shapes
: b
AllValuesQuantize/Maximum_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
AllValuesQuantize/Maximum_1MaximumAllValuesQuantize/Maximum:z:0&AllValuesQuantize/Maximum_1/y:output:0*
T0*
_output_shapes
: ï
#AllValuesQuantize/AssignMinAllValueAssignVariableOp1allvaluesquantize_minimum_readvariableop_resourceAllValuesQuantize/Minimum_1:z:0)^AllValuesQuantize/Minimum/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(ï
#AllValuesQuantize/AssignMaxAllValueAssignVariableOp1allvaluesquantize_maximum_readvariableop_resourceAllValuesQuantize/Maximum_1:z:0)^AllValuesQuantize/Maximum/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(È
8AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp1allvaluesquantize_minimum_readvariableop_resource$^AllValuesQuantize/AssignMinAllValue*
_output_shapes
: *
dtype0Ê
:AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp1allvaluesquantize_maximum_readvariableop_resource$^AllValuesQuantize/AssignMaxAllValue*
_output_shapes
: *
dtype0
)AllValuesQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsinputs@AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BAllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿææ
IdentityIdentity3AllValuesQuantize/FakeQuantWithMinMaxVars:outputs:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿææà
NoOpNoOp$^AllValuesQuantize/AssignMaxAllValue$^AllValuesQuantize/AssignMinAllValue9^AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1)^AllValuesQuantize/Maximum/ReadVariableOp)^AllValuesQuantize/Minimum/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿææ: : 2J
#AllValuesQuantize/AssignMaxAllValue#AllValuesQuantize/AssignMaxAllValue2J
#AllValuesQuantize/AssignMinAllValue#AllValuesQuantize/AssignMinAllValue2t
8AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12T
(AllValuesQuantize/Maximum/ReadVariableOp(AllValuesQuantize/Maximum/ReadVariableOp2T
(AllValuesQuantize/Minimum/ReadVariableOp(AllValuesQuantize/Minimum/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿææ
 
_user_specified_nameinputs
ÿ)
Û
J__inference_quant_conv2d_1_layer_call_and_return_conditional_losses_146278

inputsb
Hlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource:  X
Jlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource: X
Jlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource: -
biasadd_readvariableop_resource: K
Amovingavgquantize_fakequantwithminmaxvars_readvariableop_resource: M
Cmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: 
identity¢BiasAdd/ReadVariableOp¢?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp¢ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1¢ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2¢8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp¢:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1¢1conv2d_1/kernel/Regularizer/L2Loss/ReadVariableOpÐ
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpReadVariableOpHlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource*&
_output_shapes
:  *
dtype0È
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ReadVariableOpJlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource*
_output_shapes
: *
dtype0È
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ReadVariableOpJlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource*
_output_shapes
: *
dtype0ð
0LastValueQuant/FakeQuantWithMinMaxVarsPerChannel!FakeQuantWithMinMaxVarsPerChannelGLastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2:value:0*&
_output_shapes
:  *
narrow_range(¶
Conv2DConv2Dinputs:LastValueQuant/FakeQuantWithMinMaxVarsPerChannel:outputs:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿKK *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿKK X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿKK ²
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpAmovingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype0¶
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpCmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0
)MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsRelu:activations:0@MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BMovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿKK Â
1conv2d_1/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpHlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource*&
_output_shapes
:  *
dtype0
"conv2d_1/kernel/Regularizer/L2LossL2Loss9conv2d_1/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!conv2d_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o; 
conv2d_1/kernel/Regularizer/mulMul*conv2d_1/kernel/Regularizer/mul/x:output:0+conv2d_1/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: 
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿKK Õ
NoOpNoOp^BiasAdd/ReadVariableOp@^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpB^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1B^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_29^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12^conv2d_1/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿKK : : : : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp2
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_12
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_22t
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12f
1conv2d_1/kernel/Regularizer/L2Loss/ReadVariableOp1conv2d_1/kernel/Regularizer/L2Loss/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿKK 
 
_user_specified_nameinputs
ö)
×
H__inference_quant_conv2d_layer_call_and_return_conditional_losses_146142

inputsb
Hlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource: X
Jlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource: X
Jlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource: -
biasadd_readvariableop_resource: K
Amovingavgquantize_fakequantwithminmaxvars_readvariableop_resource: M
Cmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: 
identity¢BiasAdd/ReadVariableOp¢?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp¢ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1¢ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2¢8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp¢:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1¢/conv2d/kernel/Regularizer/L2Loss/ReadVariableOpÐ
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpReadVariableOpHlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource*&
_output_shapes
: *
dtype0È
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ReadVariableOpJlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource*
_output_shapes
: *
dtype0È
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ReadVariableOpJlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource*
_output_shapes
: *
dtype0ð
0LastValueQuant/FakeQuantWithMinMaxVarsPerChannel!FakeQuantWithMinMaxVarsPerChannelGLastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2:value:0*&
_output_shapes
: *
narrow_range(¹
Conv2DConv2Dinputs:LastValueQuant/FakeQuantWithMinMaxVarsPerChannel:outputs:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿââ *
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿââ Z
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿââ ²
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpAmovingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype0¶
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpCmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0
)MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsRelu:activations:0@MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BMovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿââ À
/conv2d/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpHlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource*&
_output_shapes
: *
dtype0
 conv2d/kernel/Regularizer/L2LossL2Loss7conv2d/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: d
conv2d/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;
conv2d/kernel/Regularizer/mulMul(conv2d/kernel/Regularizer/mul/x:output:0)conv2d/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: 
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿââ Ó
NoOpNoOp^BiasAdd/ReadVariableOp@^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpB^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1B^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_29^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_10^conv2d/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):ÿÿÿÿÿÿÿÿÿææ: : : : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp2
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_12
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_22t
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12b
/conv2d/kernel/Regularizer/L2Loss/ReadVariableOp/conv2d/kernel/Regularizer/L2Loss/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿææ
 
_user_specified_nameinputs

i
0__inference_quant_dropout_1_layer_call_fn_147014

inputs
identity¢StatefulPartitionedCallÕ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *;
config_proto+)

CPU

GPU2*0J
	
   E 8 *T
fORM
K__inference_quant_dropout_1_layer_call_and_return_conditional_losses_143543p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

g
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_147173

inputs
identity¢
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

ý
J__inference_quantize_layer_layer_call_and_return_conditional_losses_146062

inputsK
Aallvaluesquantize_fakequantwithminmaxvars_readvariableop_resource: M
Callvaluesquantize_fakequantwithminmaxvars_readvariableop_1_resource: 
identity¢8AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp¢:AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1²
8AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpAallvaluesquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype0¶
:AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpCallvaluesquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0
)AllValuesQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsinputs@AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BAllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿææ
IdentityIdentity3AllValuesQuantize/FakeQuantWithMinMaxVars:outputs:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿææ¾
NoOpNoOp9^AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿææ: : 2t
8AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿææ
 
_user_specified_nameinputs
ô

/__inference_quantize_layer_layer_call_fn_146053

inputs
unknown: 
	unknown_0: 
identity¢StatefulPartitionedCalló
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿææ* 
_read_only_resource_inputs
 *;
config_proto+)

CPU

GPU2*0J
	
   E 8 *S
fNRL
J__inference_quantize_layer_layer_call_and_return_conditional_losses_144283y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿææ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿææ: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿææ
 
_user_specified_nameinputs

à

+__inference_sequential_layer_call_fn_144752
conv2d_input
unknown: 
	unknown_0: #
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: 
	unknown_5: 
	unknown_6: #
	unknown_7:  
	unknown_8: 
	unknown_9: 

unknown_10: 

unknown_11: 

unknown_12: $

unknown_13: @

unknown_14:@

unknown_15:@

unknown_16:@

unknown_17: 

unknown_18: $

unknown_19:@@

unknown_20:@

unknown_21:@

unknown_22:@

unknown_23: 

unknown_24: $

unknown_25:@ 

unknown_26: 

unknown_27: 

unknown_28: 

unknown_29: 

unknown_30: 

unknown_31:


unknown_32: 

unknown_33: 

unknown_34:	

unknown_35: 

unknown_36: 

unknown_37:


unknown_38: 

unknown_39: 

unknown_40:	

unknown_41: 

unknown_42: 

unknown_43:	

unknown_44: 

unknown_45: 

unknown_46:

unknown_47: 

unknown_48: 
identity¢StatefulPartitionedCallä
StatefulPartitionedCallStatefulPartitionedCallconv2d_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48*>
Tin7
523*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*2
_read_only_resource_inputs
	!$'*-0*;
config_proto+)

CPU

GPU2*0J
	
   E 8 *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_144544o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:ÿÿÿÿÿÿÿÿÿææ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿææ
&
_user_specified_nameconv2d_input
å
R
6__inference_quant_max_pooling2d_1_layer_call_fn_146345

inputs
identityÒ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *;
config_proto+)

CPU

GPU2*0J
	
   E 8 *Z
fURS
Q__inference_quant_max_pooling2d_1_layer_call_and_return_conditional_losses_144055h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿKK :W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿKK 
 
_user_specified_nameinputs
¯
k
O__inference_quant_max_pooling2d_layer_call_and_return_conditional_losses_143000

inputs
identity
MaxPoolMaxPoolinputs*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿKK *
ksize
*
paddingVALID*
strides
`
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿKK "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿââ :Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿââ 
 
_user_specified_nameinputs
­
m
Q__inference_quant_max_pooling2d_2_layer_call_and_return_conditional_losses_146718

inputs
identity
MaxPoolMaxPoolinputs*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
ksize
*
paddingVALID*
strides
`
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ :W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
­
m
Q__inference_quant_max_pooling2d_1_layer_call_and_return_conditional_losses_146355

inputs
identity
MaxPoolMaxPoolinputs*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
ksize
*
paddingVALID*
strides
`
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿKK :W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿKK 
 
_user_specified_nameinputs
	

/__inference_quant_conv2d_2_layer_call_fn_146389

inputs!
unknown: @
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3: 
	unknown_4: 
identity¢StatefulPartitionedCall¥
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*;
config_proto+)

CPU

GPU2*0J
	
   E 8 *S
fNRL
J__inference_quant_conv2d_2_layer_call_and_return_conditional_losses_144027w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
â
i
K__inference_quant_dropout_1_layer_call_and_return_conditional_losses_143270

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
­X
º	
J__inference_quant_conv2d_2_layer_call_and_return_conditional_losses_146467

inputsI
/lastvaluequant_batchmin_readvariableop_resource: @3
%lastvaluequant_assignminlast_resource:@3
%lastvaluequant_assignmaxlast_resource:@-
biasadd_readvariableop_resource:@@
6movingavgquantize_assignminema_readvariableop_resource: @
6movingavgquantize_assignmaxema_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢LastValueQuant/AssignMaxLast¢LastValueQuant/AssignMinLast¢&LastValueQuant/BatchMax/ReadVariableOp¢&LastValueQuant/BatchMin/ReadVariableOp¢?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp¢ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1¢ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2¢2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp¢-MovingAvgQuantize/AssignMaxEma/ReadVariableOp¢2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp¢-MovingAvgQuantize/AssignMinEma/ReadVariableOp¢8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp¢:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1¢1conv2d_2/kernel/Regularizer/L2Loss/ReadVariableOp
&LastValueQuant/BatchMin/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
: @*
dtype0~
)LastValueQuant/BatchMin/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          §
LastValueQuant/BatchMinMin.LastValueQuant/BatchMin/ReadVariableOp:value:02LastValueQuant/BatchMin/reduction_indices:output:0*
T0*
_output_shapes
:@
&LastValueQuant/BatchMax/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
: @*
dtype0~
)LastValueQuant/BatchMax/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          §
LastValueQuant/BatchMaxMax.LastValueQuant/BatchMax/ReadVariableOp:value:02LastValueQuant/BatchMax/reduction_indices:output:0*
T0*
_output_shapes
:@]
LastValueQuant/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ¿
LastValueQuant/truedivRealDiv LastValueQuant/BatchMax:output:0!LastValueQuant/truediv/y:output:0*
T0*
_output_shapes
:@
LastValueQuant/MinimumMinimum LastValueQuant/BatchMin:output:0LastValueQuant/truediv:z:0*
T0*
_output_shapes
:@Y
LastValueQuant/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ¿
LastValueQuant/mulMul LastValueQuant/BatchMin:output:0LastValueQuant/mul/y:output:0*
T0*
_output_shapes
:@
LastValueQuant/MaximumMaximum LastValueQuant/BatchMax:output:0LastValueQuant/mul:z:0*
T0*
_output_shapes
:@¬
LastValueQuant/AssignMinLastAssignVariableOp%lastvaluequant_assignminlast_resourceLastValueQuant/Minimum:z:0*
_output_shapes
 *
dtype0*
validate_shape(¬
LastValueQuant/AssignMaxLastAssignVariableOp%lastvaluequant_assignmaxlast_resourceLastValueQuant/Maximum:z:0*
_output_shapes
 *
dtype0*
validate_shape(·
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
: @*
dtype0Â
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ReadVariableOp%lastvaluequant_assignminlast_resource^LastValueQuant/AssignMinLast*
_output_shapes
:@*
dtype0Â
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ReadVariableOp%lastvaluequant_assignmaxlast_resource^LastValueQuant/AssignMaxLast*
_output_shapes
:@*
dtype0ð
0LastValueQuant/FakeQuantWithMinMaxVarsPerChannel!FakeQuantWithMinMaxVarsPerChannelGLastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2:value:0*&
_output_shapes
: @*
narrow_range(¶
Conv2DConv2Dinputs:LastValueQuant/FakeQuantWithMinMaxVarsPerChannel:outputs:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@p
MovingAvgQuantize/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             x
MovingAvgQuantize/BatchMinMinRelu:activations:0 MovingAvgQuantize/Const:output:0*
T0*
_output_shapes
: r
MovingAvgQuantize/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             z
MovingAvgQuantize/BatchMaxMaxRelu:activations:0"MovingAvgQuantize/Const_1:output:0*
T0*
_output_shapes
: `
MovingAvgQuantize/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
MovingAvgQuantize/MinimumMinimum#MovingAvgQuantize/BatchMin:output:0$MovingAvgQuantize/Minimum/y:output:0*
T0*
_output_shapes
: `
MovingAvgQuantize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
MovingAvgQuantize/MaximumMaximum#MovingAvgQuantize/BatchMax:output:0$MovingAvgQuantize/Maximum/y:output:0*
T0*
_output_shapes
: i
$MovingAvgQuantize/AssignMinEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o:
-MovingAvgQuantize/AssignMinEma/ReadVariableOpReadVariableOp6movingavgquantize_assignminema_readvariableop_resource*
_output_shapes
: *
dtype0 
"MovingAvgQuantize/AssignMinEma/subSub5MovingAvgQuantize/AssignMinEma/ReadVariableOp:value:0MovingAvgQuantize/Minimum:z:0*
T0*
_output_shapes
: ¡
"MovingAvgQuantize/AssignMinEma/mulMul&MovingAvgQuantize/AssignMinEma/sub:z:0-MovingAvgQuantize/AssignMinEma/decay:output:0*
T0*
_output_shapes
: ü
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOpAssignSubVariableOp6movingavgquantize_assignminema_readvariableop_resource&MovingAvgQuantize/AssignMinEma/mul:z:0.^MovingAvgQuantize/AssignMinEma/ReadVariableOp*
_output_shapes
 *
dtype0i
$MovingAvgQuantize/AssignMaxEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o:
-MovingAvgQuantize/AssignMaxEma/ReadVariableOpReadVariableOp6movingavgquantize_assignmaxema_readvariableop_resource*
_output_shapes
: *
dtype0 
"MovingAvgQuantize/AssignMaxEma/subSub5MovingAvgQuantize/AssignMaxEma/ReadVariableOp:value:0MovingAvgQuantize/Maximum:z:0*
T0*
_output_shapes
: ¡
"MovingAvgQuantize/AssignMaxEma/mulMul&MovingAvgQuantize/AssignMaxEma/sub:z:0-MovingAvgQuantize/AssignMaxEma/decay:output:0*
T0*
_output_shapes
: ü
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpAssignSubVariableOp6movingavgquantize_assignmaxema_readvariableop_resource&MovingAvgQuantize/AssignMaxEma/mul:z:0.^MovingAvgQuantize/AssignMaxEma/ReadVariableOp*
_output_shapes
 *
dtype0Ü
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp6movingavgquantize_assignminema_readvariableop_resource3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp*
_output_shapes
: *
dtype0Þ
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp6movingavgquantize_assignmaxema_readvariableop_resource3^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp*
_output_shapes
: *
dtype0
)MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsRelu:activations:0@MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BMovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@©
1conv2d_2/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
: @*
dtype0
"conv2d_2/kernel/Regularizer/L2LossL2Loss9conv2d_2/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!conv2d_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o; 
conv2d_2/kernel/Regularizer/mulMul*conv2d_2/kernel/Regularizer/mul/x:output:0+conv2d_2/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: 
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¯
NoOpNoOp^BiasAdd/ReadVariableOp^LastValueQuant/AssignMaxLast^LastValueQuant/AssignMinLast'^LastValueQuant/BatchMax/ReadVariableOp'^LastValueQuant/BatchMin/ReadVariableOp@^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpB^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1B^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_23^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp.^MovingAvgQuantize/AssignMaxEma/ReadVariableOp3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp.^MovingAvgQuantize/AssignMinEma/ReadVariableOp9^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12^conv2d_2/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ : : : : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2<
LastValueQuant/AssignMaxLastLastValueQuant/AssignMaxLast2<
LastValueQuant/AssignMinLastLastValueQuant/AssignMinLast2P
&LastValueQuant/BatchMax/ReadVariableOp&LastValueQuant/BatchMax/ReadVariableOp2P
&LastValueQuant/BatchMin/ReadVariableOp&LastValueQuant/BatchMin/ReadVariableOp2
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp2
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_12
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_22h
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2^
-MovingAvgQuantize/AssignMaxEma/ReadVariableOp-MovingAvgQuantize/AssignMaxEma/ReadVariableOp2h
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2^
-MovingAvgQuantize/AssignMinEma/ReadVariableOp-MovingAvgQuantize/AssignMinEma/ReadVariableOp2t
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12f
1conv2d_2/kernel/Regularizer/L2Loss/ReadVariableOp1conv2d_2/kernel/Regularizer/L2Loss/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
­
m
Q__inference_quant_max_pooling2d_1_layer_call_and_return_conditional_losses_146350

inputs
identity
MaxPoolMaxPoolinputs*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
ksize
*
paddingVALID*
strides
`
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿKK :W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿKK 
 
_user_specified_nameinputs

e
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_142907

inputs
identity¢
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
±À
Ñ.
__inference__traced_save_147569
file_prefix@
<savev2_quantize_layer_quantize_layer_min_read_readvariableop@
<savev2_quantize_layer_quantize_layer_max_read_readvariableop<
8savev2_quantize_layer_optimizer_step_read_readvariableop:
6savev2_quant_conv2d_optimizer_step_read_readvariableop6
2savev2_quant_conv2d_kernel_min_read_readvariableop6
2savev2_quant_conv2d_kernel_max_read_readvariableop?
;savev2_quant_conv2d_post_activation_min_read_readvariableop?
;savev2_quant_conv2d_post_activation_max_read_readvariableopA
=savev2_quant_max_pooling2d_optimizer_step_read_readvariableop<
8savev2_quant_conv2d_1_optimizer_step_read_readvariableop8
4savev2_quant_conv2d_1_kernel_min_read_readvariableop8
4savev2_quant_conv2d_1_kernel_max_read_readvariableopA
=savev2_quant_conv2d_1_post_activation_min_read_readvariableopA
=savev2_quant_conv2d_1_post_activation_max_read_readvariableopC
?savev2_quant_max_pooling2d_1_optimizer_step_read_readvariableop<
8savev2_quant_conv2d_2_optimizer_step_read_readvariableop8
4savev2_quant_conv2d_2_kernel_min_read_readvariableop8
4savev2_quant_conv2d_2_kernel_max_read_readvariableopA
=savev2_quant_conv2d_2_post_activation_min_read_readvariableopA
=savev2_quant_conv2d_2_post_activation_max_read_readvariableop<
8savev2_quant_conv2d_3_optimizer_step_read_readvariableop8
4savev2_quant_conv2d_3_kernel_min_read_readvariableop8
4savev2_quant_conv2d_3_kernel_max_read_readvariableopA
=savev2_quant_conv2d_3_post_activation_min_read_readvariableopA
=savev2_quant_conv2d_3_post_activation_max_read_readvariableop<
8savev2_quant_conv2d_4_optimizer_step_read_readvariableop8
4savev2_quant_conv2d_4_kernel_min_read_readvariableop8
4savev2_quant_conv2d_4_kernel_max_read_readvariableopA
=savev2_quant_conv2d_4_post_activation_min_read_readvariableopA
=savev2_quant_conv2d_4_post_activation_max_read_readvariableopC
?savev2_quant_max_pooling2d_2_optimizer_step_read_readvariableop;
7savev2_quant_flatten_optimizer_step_read_readvariableop9
5savev2_quant_dense_optimizer_step_read_readvariableop5
1savev2_quant_dense_kernel_min_read_readvariableop5
1savev2_quant_dense_kernel_max_read_readvariableop>
:savev2_quant_dense_post_activation_min_read_readvariableop>
:savev2_quant_dense_post_activation_max_read_readvariableop;
7savev2_quant_dropout_optimizer_step_read_readvariableop;
7savev2_quant_dense_1_optimizer_step_read_readvariableop7
3savev2_quant_dense_1_kernel_min_read_readvariableop7
3savev2_quant_dense_1_kernel_max_read_readvariableop@
<savev2_quant_dense_1_post_activation_min_read_readvariableop@
<savev2_quant_dense_1_post_activation_max_read_readvariableop=
9savev2_quant_dropout_1_optimizer_step_read_readvariableop;
7savev2_quant_dense_2_optimizer_step_read_readvariableop7
3savev2_quant_dense_2_kernel_min_read_readvariableop7
3savev2_quant_dense_2_kernel_max_read_readvariableop?
;savev2_quant_dense_2_pre_activation_min_read_readvariableop?
;savev2_quant_dense_2_pre_activation_max_read_readvariableop,
(savev2_conv2d_kernel_read_readvariableop*
&savev2_conv2d_bias_read_readvariableop.
*savev2_conv2d_1_kernel_read_readvariableop,
(savev2_conv2d_1_bias_read_readvariableop.
*savev2_conv2d_2_kernel_read_readvariableop,
(savev2_conv2d_2_bias_read_readvariableop.
*savev2_conv2d_3_kernel_read_readvariableop,
(savev2_conv2d_3_bias_read_readvariableop.
*savev2_conv2d_4_kernel_read_readvariableop,
(savev2_conv2d_4_bias_read_readvariableop+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop-
)savev2_dense_2_kernel_read_readvariableop+
'savev2_dense_2_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop3
/savev2_adam_conv2d_kernel_m_read_readvariableop1
-savev2_adam_conv2d_bias_m_read_readvariableop5
1savev2_adam_conv2d_1_kernel_m_read_readvariableop3
/savev2_adam_conv2d_1_bias_m_read_readvariableop5
1savev2_adam_conv2d_2_kernel_m_read_readvariableop3
/savev2_adam_conv2d_2_bias_m_read_readvariableop5
1savev2_adam_conv2d_3_kernel_m_read_readvariableop3
/savev2_adam_conv2d_3_bias_m_read_readvariableop5
1savev2_adam_conv2d_4_kernel_m_read_readvariableop3
/savev2_adam_conv2d_4_bias_m_read_readvariableop2
.savev2_adam_dense_kernel_m_read_readvariableop0
,savev2_adam_dense_bias_m_read_readvariableop4
0savev2_adam_dense_1_kernel_m_read_readvariableop2
.savev2_adam_dense_1_bias_m_read_readvariableop4
0savev2_adam_dense_2_kernel_m_read_readvariableop2
.savev2_adam_dense_2_bias_m_read_readvariableop3
/savev2_adam_conv2d_kernel_v_read_readvariableop1
-savev2_adam_conv2d_bias_v_read_readvariableop5
1savev2_adam_conv2d_1_kernel_v_read_readvariableop3
/savev2_adam_conv2d_1_bias_v_read_readvariableop5
1savev2_adam_conv2d_2_kernel_v_read_readvariableop3
/savev2_adam_conv2d_2_bias_v_read_readvariableop5
1savev2_adam_conv2d_3_kernel_v_read_readvariableop3
/savev2_adam_conv2d_3_bias_v_read_readvariableop5
1savev2_adam_conv2d_4_kernel_v_read_readvariableop3
/savev2_adam_conv2d_4_bias_v_read_readvariableop2
.savev2_adam_dense_kernel_v_read_readvariableop0
,savev2_adam_dense_bias_v_read_readvariableop4
0savev2_adam_dense_1_kernel_v_read_readvariableop2
.savev2_adam_dense_1_bias_v_read_readvariableop4
0savev2_adam_dense_2_kernel_v_read_readvariableop2
.savev2_adam_dense_2_bias_v_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: Ø3
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:k*
dtype0*3
value÷2Bô2kBBlayer_with_weights-0/quantize_layer_min/.ATTRIBUTES/VARIABLE_VALUEBBlayer_with_weights-0/quantize_layer_max/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-0/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-1/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-1/kernel_min/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-1/kernel_max/.ATTRIBUTES/VARIABLE_VALUEBClayer_with_weights-1/post_activation_min/.ATTRIBUTES/VARIABLE_VALUEBClayer_with_weights-1/post_activation_max/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-2/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-3/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-3/kernel_min/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-3/kernel_max/.ATTRIBUTES/VARIABLE_VALUEBClayer_with_weights-3/post_activation_min/.ATTRIBUTES/VARIABLE_VALUEBClayer_with_weights-3/post_activation_max/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-4/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-5/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-5/kernel_min/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-5/kernel_max/.ATTRIBUTES/VARIABLE_VALUEBClayer_with_weights-5/post_activation_min/.ATTRIBUTES/VARIABLE_VALUEBClayer_with_weights-5/post_activation_max/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-6/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-6/kernel_min/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-6/kernel_max/.ATTRIBUTES/VARIABLE_VALUEBClayer_with_weights-6/post_activation_min/.ATTRIBUTES/VARIABLE_VALUEBClayer_with_weights-6/post_activation_max/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-7/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-7/kernel_min/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-7/kernel_max/.ATTRIBUTES/VARIABLE_VALUEBClayer_with_weights-7/post_activation_min/.ATTRIBUTES/VARIABLE_VALUEBClayer_with_weights-7/post_activation_max/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-8/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-9/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-10/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-10/kernel_min/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-10/kernel_max/.ATTRIBUTES/VARIABLE_VALUEBDlayer_with_weights-10/post_activation_min/.ATTRIBUTES/VARIABLE_VALUEBDlayer_with_weights-10/post_activation_max/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-11/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-12/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-12/kernel_min/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-12/kernel_max/.ATTRIBUTES/VARIABLE_VALUEBDlayer_with_weights-12/post_activation_min/.ATTRIBUTES/VARIABLE_VALUEBDlayer_with_weights-12/post_activation_max/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-13/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-14/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-14/kernel_min/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-14/kernel_max/.ATTRIBUTES/VARIABLE_VALUEBClayer_with_weights-14/pre_activation_min/.ATTRIBUTES/VARIABLE_VALUEBClayer_with_weights-14/pre_activation_max/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/26/.ATTRIBUTES/VARIABLE_VALUEB'variables/27/.ATTRIBUTES/VARIABLE_VALUEB'variables/33/.ATTRIBUTES/VARIABLE_VALUEB'variables/34/.ATTRIBUTES/VARIABLE_VALUEB'variables/42/.ATTRIBUTES/VARIABLE_VALUEB'variables/43/.ATTRIBUTES/VARIABLE_VALUEB'variables/50/.ATTRIBUTES/VARIABLE_VALUEB'variables/51/.ATTRIBUTES/VARIABLE_VALUEB'variables/58/.ATTRIBUTES/VARIABLE_VALUEB'variables/59/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/19/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/20/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/26/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/27/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/33/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/34/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/42/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/43/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/50/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/51/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/58/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/59/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/19/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/20/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/26/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/27/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/33/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/34/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/42/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/43/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/50/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/51/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/58/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/59/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHÆ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:k*
dtype0*ë
valueáBÞkB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ×,
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0<savev2_quantize_layer_quantize_layer_min_read_readvariableop<savev2_quantize_layer_quantize_layer_max_read_readvariableop8savev2_quantize_layer_optimizer_step_read_readvariableop6savev2_quant_conv2d_optimizer_step_read_readvariableop2savev2_quant_conv2d_kernel_min_read_readvariableop2savev2_quant_conv2d_kernel_max_read_readvariableop;savev2_quant_conv2d_post_activation_min_read_readvariableop;savev2_quant_conv2d_post_activation_max_read_readvariableop=savev2_quant_max_pooling2d_optimizer_step_read_readvariableop8savev2_quant_conv2d_1_optimizer_step_read_readvariableop4savev2_quant_conv2d_1_kernel_min_read_readvariableop4savev2_quant_conv2d_1_kernel_max_read_readvariableop=savev2_quant_conv2d_1_post_activation_min_read_readvariableop=savev2_quant_conv2d_1_post_activation_max_read_readvariableop?savev2_quant_max_pooling2d_1_optimizer_step_read_readvariableop8savev2_quant_conv2d_2_optimizer_step_read_readvariableop4savev2_quant_conv2d_2_kernel_min_read_readvariableop4savev2_quant_conv2d_2_kernel_max_read_readvariableop=savev2_quant_conv2d_2_post_activation_min_read_readvariableop=savev2_quant_conv2d_2_post_activation_max_read_readvariableop8savev2_quant_conv2d_3_optimizer_step_read_readvariableop4savev2_quant_conv2d_3_kernel_min_read_readvariableop4savev2_quant_conv2d_3_kernel_max_read_readvariableop=savev2_quant_conv2d_3_post_activation_min_read_readvariableop=savev2_quant_conv2d_3_post_activation_max_read_readvariableop8savev2_quant_conv2d_4_optimizer_step_read_readvariableop4savev2_quant_conv2d_4_kernel_min_read_readvariableop4savev2_quant_conv2d_4_kernel_max_read_readvariableop=savev2_quant_conv2d_4_post_activation_min_read_readvariableop=savev2_quant_conv2d_4_post_activation_max_read_readvariableop?savev2_quant_max_pooling2d_2_optimizer_step_read_readvariableop7savev2_quant_flatten_optimizer_step_read_readvariableop5savev2_quant_dense_optimizer_step_read_readvariableop1savev2_quant_dense_kernel_min_read_readvariableop1savev2_quant_dense_kernel_max_read_readvariableop:savev2_quant_dense_post_activation_min_read_readvariableop:savev2_quant_dense_post_activation_max_read_readvariableop7savev2_quant_dropout_optimizer_step_read_readvariableop7savev2_quant_dense_1_optimizer_step_read_readvariableop3savev2_quant_dense_1_kernel_min_read_readvariableop3savev2_quant_dense_1_kernel_max_read_readvariableop<savev2_quant_dense_1_post_activation_min_read_readvariableop<savev2_quant_dense_1_post_activation_max_read_readvariableop9savev2_quant_dropout_1_optimizer_step_read_readvariableop7savev2_quant_dense_2_optimizer_step_read_readvariableop3savev2_quant_dense_2_kernel_min_read_readvariableop3savev2_quant_dense_2_kernel_max_read_readvariableop;savev2_quant_dense_2_pre_activation_min_read_readvariableop;savev2_quant_dense_2_pre_activation_max_read_readvariableop(savev2_conv2d_kernel_read_readvariableop&savev2_conv2d_bias_read_readvariableop*savev2_conv2d_1_kernel_read_readvariableop(savev2_conv2d_1_bias_read_readvariableop*savev2_conv2d_2_kernel_read_readvariableop(savev2_conv2d_2_bias_read_readvariableop*savev2_conv2d_3_kernel_read_readvariableop(savev2_conv2d_3_bias_read_readvariableop*savev2_conv2d_4_kernel_read_readvariableop(savev2_conv2d_4_bias_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop)savev2_dense_2_kernel_read_readvariableop'savev2_dense_2_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop/savev2_adam_conv2d_kernel_m_read_readvariableop-savev2_adam_conv2d_bias_m_read_readvariableop1savev2_adam_conv2d_1_kernel_m_read_readvariableop/savev2_adam_conv2d_1_bias_m_read_readvariableop1savev2_adam_conv2d_2_kernel_m_read_readvariableop/savev2_adam_conv2d_2_bias_m_read_readvariableop1savev2_adam_conv2d_3_kernel_m_read_readvariableop/savev2_adam_conv2d_3_bias_m_read_readvariableop1savev2_adam_conv2d_4_kernel_m_read_readvariableop/savev2_adam_conv2d_4_bias_m_read_readvariableop.savev2_adam_dense_kernel_m_read_readvariableop,savev2_adam_dense_bias_m_read_readvariableop0savev2_adam_dense_1_kernel_m_read_readvariableop.savev2_adam_dense_1_bias_m_read_readvariableop0savev2_adam_dense_2_kernel_m_read_readvariableop.savev2_adam_dense_2_bias_m_read_readvariableop/savev2_adam_conv2d_kernel_v_read_readvariableop-savev2_adam_conv2d_bias_v_read_readvariableop1savev2_adam_conv2d_1_kernel_v_read_readvariableop/savev2_adam_conv2d_1_bias_v_read_readvariableop1savev2_adam_conv2d_2_kernel_v_read_readvariableop/savev2_adam_conv2d_2_bias_v_read_readvariableop1savev2_adam_conv2d_3_kernel_v_read_readvariableop/savev2_adam_conv2d_3_bias_v_read_readvariableop1savev2_adam_conv2d_4_kernel_v_read_readvariableop/savev2_adam_conv2d_4_bias_v_read_readvariableop.savev2_adam_dense_kernel_v_read_readvariableop,savev2_adam_dense_bias_v_read_readvariableop0savev2_adam_dense_1_kernel_v_read_readvariableop.savev2_adam_dense_1_bias_v_read_readvariableop0savev2_adam_dense_2_kernel_v_read_readvariableop.savev2_adam_dense_2_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *y
dtypeso
m2k	
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*Â
_input_shapes°
­: : : : : : : : : : : : : : : : : :@:@: : : :@:@: : : : : : : : : : : : : : : : : : : : : : : : : : : : :  : : @:@:@@:@:@ : :
::
::	:: : : : : : : : : : : :  : : @:@:@@:@:@ : :
::
::	:: : :  : : @:@:@@:@:@ : :
::
::	:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
:@: 

_output_shapes
:@:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
:@: 

_output_shapes
:@:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
: :!

_output_shapes
: :"

_output_shapes
: :#

_output_shapes
: :$

_output_shapes
: :%

_output_shapes
: :&

_output_shapes
: :'

_output_shapes
: :(

_output_shapes
: :)

_output_shapes
: :*

_output_shapes
: :+

_output_shapes
: :,

_output_shapes
: :-

_output_shapes
: :.

_output_shapes
: :/

_output_shapes
: :0

_output_shapes
: :1

_output_shapes
: :,2(
&
_output_shapes
: : 3

_output_shapes
: :,4(
&
_output_shapes
:  : 5

_output_shapes
: :,6(
&
_output_shapes
: @: 7

_output_shapes
:@:,8(
&
_output_shapes
:@@: 9

_output_shapes
:@:,:(
&
_output_shapes
:@ : ;

_output_shapes
: :&<"
 
_output_shapes
:
:!=

_output_shapes	
::&>"
 
_output_shapes
:
:!?

_output_shapes	
::%@!

_output_shapes
:	: A

_output_shapes
::B

_output_shapes
: :C

_output_shapes
: :D

_output_shapes
: :E

_output_shapes
: :F

_output_shapes
: :G

_output_shapes
: :H

_output_shapes
: :I

_output_shapes
: :J

_output_shapes
: :,K(
&
_output_shapes
: : L

_output_shapes
: :,M(
&
_output_shapes
:  : N

_output_shapes
: :,O(
&
_output_shapes
: @: P

_output_shapes
:@:,Q(
&
_output_shapes
:@@: R

_output_shapes
:@:,S(
&
_output_shapes
:@ : T

_output_shapes
: :&U"
 
_output_shapes
:
:!V

_output_shapes	
::&W"
 
_output_shapes
:
:!X

_output_shapes	
::%Y!

_output_shapes
:	: Z

_output_shapes
::,[(
&
_output_shapes
: : \

_output_shapes
: :,](
&
_output_shapes
:  : ^

_output_shapes
: :,_(
&
_output_shapes
: @: `

_output_shapes
:@:,a(
&
_output_shapes
:@@: b

_output_shapes
:@:,c(
&
_output_shapes
:@ : d

_output_shapes
: :&e"
 
_output_shapes
:
:!f

_output_shapes	
::&g"
 
_output_shapes
:
:!h

_output_shapes	
::%i!

_output_shapes
:	: j

_output_shapes
::k

_output_shapes
: 
Ç
J
.__inference_quant_flatten_layer_call_fn_146728

inputs
identityÃ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *;
config_proto+)

CPU

GPU2*0J
	
   E 8 *R
fMRK
I__inference_quant_flatten_layer_call_and_return_conditional_losses_143178a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ :W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
¯
k
O__inference_quant_max_pooling2d_layer_call_and_return_conditional_losses_144159

inputs
identity
MaxPoolMaxPoolinputs*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿKK *
ksize
*
paddingVALID*
strides
`
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿKK "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿââ :Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿââ 
 
_user_specified_nameinputs

e
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_147154

inputs
identity¢
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

ý
J__inference_quantize_layer_layer_call_and_return_conditional_losses_142950

inputsK
Aallvaluesquantize_fakequantwithminmaxvars_readvariableop_resource: M
Callvaluesquantize_fakequantwithminmaxvars_readvariableop_1_resource: 
identity¢8AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp¢:AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1²
8AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpAallvaluesquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype0¶
:AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpCallvaluesquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0
)AllValuesQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsinputs@AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BAllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿææ
IdentityIdentity3AllValuesQuantize/FakeQuantWithMinMaxVars:outputs:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿææ¾
NoOpNoOp9^AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿææ: : 2t
8AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿææ
 
_user_specified_nameinputs
	
þ
-__inference_quant_conv2d_layer_call_fn_146117

inputs!
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: 
identity¢StatefulPartitionedCall¥
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿââ *$
_read_only_resource_inputs
*;
config_proto+)

CPU

GPU2*0J
	
   E 8 *Q
fLRJ
H__inference_quant_conv2d_layer_call_and_return_conditional_losses_144235y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿââ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):ÿÿÿÿÿÿÿÿÿææ: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿææ
 
_user_specified_nameinputs
­X
º	
J__inference_quant_conv2d_3_layer_call_and_return_conditional_losses_143939

inputsI
/lastvaluequant_batchmin_readvariableop_resource:@@3
%lastvaluequant_assignminlast_resource:@3
%lastvaluequant_assignmaxlast_resource:@-
biasadd_readvariableop_resource:@@
6movingavgquantize_assignminema_readvariableop_resource: @
6movingavgquantize_assignmaxema_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢LastValueQuant/AssignMaxLast¢LastValueQuant/AssignMinLast¢&LastValueQuant/BatchMax/ReadVariableOp¢&LastValueQuant/BatchMin/ReadVariableOp¢?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp¢ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1¢ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2¢2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp¢-MovingAvgQuantize/AssignMaxEma/ReadVariableOp¢2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp¢-MovingAvgQuantize/AssignMinEma/ReadVariableOp¢8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp¢:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1¢1conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOp
&LastValueQuant/BatchMin/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:@@*
dtype0~
)LastValueQuant/BatchMin/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          §
LastValueQuant/BatchMinMin.LastValueQuant/BatchMin/ReadVariableOp:value:02LastValueQuant/BatchMin/reduction_indices:output:0*
T0*
_output_shapes
:@
&LastValueQuant/BatchMax/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:@@*
dtype0~
)LastValueQuant/BatchMax/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          §
LastValueQuant/BatchMaxMax.LastValueQuant/BatchMax/ReadVariableOp:value:02LastValueQuant/BatchMax/reduction_indices:output:0*
T0*
_output_shapes
:@]
LastValueQuant/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ¿
LastValueQuant/truedivRealDiv LastValueQuant/BatchMax:output:0!LastValueQuant/truediv/y:output:0*
T0*
_output_shapes
:@
LastValueQuant/MinimumMinimum LastValueQuant/BatchMin:output:0LastValueQuant/truediv:z:0*
T0*
_output_shapes
:@Y
LastValueQuant/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ¿
LastValueQuant/mulMul LastValueQuant/BatchMin:output:0LastValueQuant/mul/y:output:0*
T0*
_output_shapes
:@
LastValueQuant/MaximumMaximum LastValueQuant/BatchMax:output:0LastValueQuant/mul:z:0*
T0*
_output_shapes
:@¬
LastValueQuant/AssignMinLastAssignVariableOp%lastvaluequant_assignminlast_resourceLastValueQuant/Minimum:z:0*
_output_shapes
 *
dtype0*
validate_shape(¬
LastValueQuant/AssignMaxLastAssignVariableOp%lastvaluequant_assignmaxlast_resourceLastValueQuant/Maximum:z:0*
_output_shapes
 *
dtype0*
validate_shape(·
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:@@*
dtype0Â
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ReadVariableOp%lastvaluequant_assignminlast_resource^LastValueQuant/AssignMinLast*
_output_shapes
:@*
dtype0Â
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ReadVariableOp%lastvaluequant_assignmaxlast_resource^LastValueQuant/AssignMaxLast*
_output_shapes
:@*
dtype0ð
0LastValueQuant/FakeQuantWithMinMaxVarsPerChannel!FakeQuantWithMinMaxVarsPerChannelGLastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2:value:0*&
_output_shapes
:@@*
narrow_range(¶
Conv2DConv2Dinputs:LastValueQuant/FakeQuantWithMinMaxVarsPerChannel:outputs:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@p
MovingAvgQuantize/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             x
MovingAvgQuantize/BatchMinMinRelu:activations:0 MovingAvgQuantize/Const:output:0*
T0*
_output_shapes
: r
MovingAvgQuantize/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             z
MovingAvgQuantize/BatchMaxMaxRelu:activations:0"MovingAvgQuantize/Const_1:output:0*
T0*
_output_shapes
: `
MovingAvgQuantize/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
MovingAvgQuantize/MinimumMinimum#MovingAvgQuantize/BatchMin:output:0$MovingAvgQuantize/Minimum/y:output:0*
T0*
_output_shapes
: `
MovingAvgQuantize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
MovingAvgQuantize/MaximumMaximum#MovingAvgQuantize/BatchMax:output:0$MovingAvgQuantize/Maximum/y:output:0*
T0*
_output_shapes
: i
$MovingAvgQuantize/AssignMinEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o:
-MovingAvgQuantize/AssignMinEma/ReadVariableOpReadVariableOp6movingavgquantize_assignminema_readvariableop_resource*
_output_shapes
: *
dtype0 
"MovingAvgQuantize/AssignMinEma/subSub5MovingAvgQuantize/AssignMinEma/ReadVariableOp:value:0MovingAvgQuantize/Minimum:z:0*
T0*
_output_shapes
: ¡
"MovingAvgQuantize/AssignMinEma/mulMul&MovingAvgQuantize/AssignMinEma/sub:z:0-MovingAvgQuantize/AssignMinEma/decay:output:0*
T0*
_output_shapes
: ü
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOpAssignSubVariableOp6movingavgquantize_assignminema_readvariableop_resource&MovingAvgQuantize/AssignMinEma/mul:z:0.^MovingAvgQuantize/AssignMinEma/ReadVariableOp*
_output_shapes
 *
dtype0i
$MovingAvgQuantize/AssignMaxEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o:
-MovingAvgQuantize/AssignMaxEma/ReadVariableOpReadVariableOp6movingavgquantize_assignmaxema_readvariableop_resource*
_output_shapes
: *
dtype0 
"MovingAvgQuantize/AssignMaxEma/subSub5MovingAvgQuantize/AssignMaxEma/ReadVariableOp:value:0MovingAvgQuantize/Maximum:z:0*
T0*
_output_shapes
: ¡
"MovingAvgQuantize/AssignMaxEma/mulMul&MovingAvgQuantize/AssignMaxEma/sub:z:0-MovingAvgQuantize/AssignMaxEma/decay:output:0*
T0*
_output_shapes
: ü
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpAssignSubVariableOp6movingavgquantize_assignmaxema_readvariableop_resource&MovingAvgQuantize/AssignMaxEma/mul:z:0.^MovingAvgQuantize/AssignMaxEma/ReadVariableOp*
_output_shapes
 *
dtype0Ü
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp6movingavgquantize_assignminema_readvariableop_resource3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp*
_output_shapes
: *
dtype0Þ
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp6movingavgquantize_assignmaxema_readvariableop_resource3^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp*
_output_shapes
: *
dtype0
)MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsRelu:activations:0@MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BMovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@©
1conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:@@*
dtype0
"conv2d_3/kernel/Regularizer/L2LossL2Loss9conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!conv2d_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o; 
conv2d_3/kernel/Regularizer/mulMul*conv2d_3/kernel/Regularizer/mul/x:output:0+conv2d_3/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: 
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¯
NoOpNoOp^BiasAdd/ReadVariableOp^LastValueQuant/AssignMaxLast^LastValueQuant/AssignMinLast'^LastValueQuant/BatchMax/ReadVariableOp'^LastValueQuant/BatchMin/ReadVariableOp@^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpB^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1B^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_23^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp.^MovingAvgQuantize/AssignMaxEma/ReadVariableOp3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp.^MovingAvgQuantize/AssignMinEma/ReadVariableOp9^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12^conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ@: : : : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2<
LastValueQuant/AssignMaxLastLastValueQuant/AssignMaxLast2<
LastValueQuant/AssignMinLastLastValueQuant/AssignMinLast2P
&LastValueQuant/BatchMax/ReadVariableOp&LastValueQuant/BatchMax/ReadVariableOp2P
&LastValueQuant/BatchMin/ReadVariableOp&LastValueQuant/BatchMin/ReadVariableOp2
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp2
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_12
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_22h
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2^
-MovingAvgQuantize/AssignMaxEma/ReadVariableOp-MovingAvgQuantize/AssignMaxEma/ReadVariableOp2h
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2^
-MovingAvgQuantize/AssignMinEma/ReadVariableOp-MovingAvgQuantize/AssignMinEma/ReadVariableOp2t
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12f
1conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOp1conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
	
µ
__inference_loss_fn_0_147144R
8conv2d_kernel_regularizer_l2loss_readvariableop_resource: 
identity¢/conv2d/kernel/Regularizer/L2Loss/ReadVariableOp°
/conv2d/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp8conv2d_kernel_regularizer_l2loss_readvariableop_resource*&
_output_shapes
: *
dtype0
 conv2d/kernel/Regularizer/L2LossL2Loss7conv2d/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: d
conv2d/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;
conv2d/kernel/Regularizer/mulMul(conv2d/kernel/Regularizer/mul/x:output:0)conv2d/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: _
IdentityIdentity!conv2d/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: x
NoOpNoOp0^conv2d/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2b
/conv2d/kernel/Regularizer/L2Loss/ReadVariableOp/conv2d/kernel/Regularizer/L2Loss/ReadVariableOp
	

/__inference_quant_conv2d_3_layer_call_fn_146505

inputs!
unknown:@@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3: 
	unknown_4: 
identity¢StatefulPartitionedCall¥
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*;
config_proto+)

CPU

GPU2*0J
	
   E 8 *S
fNRL
J__inference_quant_conv2d_3_layer_call_and_return_conditional_losses_143939w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ@: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
à
g
I__inference_quant_dropout_layer_call_and_return_conditional_losses_146876

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
­
m
Q__inference_quant_max_pooling2d_1_layer_call_and_return_conditional_losses_143046

inputs
identity
MaxPoolMaxPoolinputs*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
ksize
*
paddingVALID*
strides
`
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿKK :W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿKK 
 
_user_specified_nameinputs
¯
k
O__inference_quant_max_pooling2d_layer_call_and_return_conditional_losses_146214

inputs
identity
MaxPoolMaxPoolinputs*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿKK *
ksize
*
paddingVALID*
strides
`
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿKK "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿââ :Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿââ 
 
_user_specified_nameinputs
Ù&

I__inference_quant_dense_1_layer_call_and_return_conditional_losses_143251

inputsR
>lastvaluequant_fakequantwithminmaxvars_readvariableop_resource:
J
@lastvaluequant_fakequantwithminmaxvars_readvariableop_1_resource: J
@lastvaluequant_fakequantwithminmaxvars_readvariableop_2_resource: .
biasadd_readvariableop_resource:	K
Amovingavgquantize_fakequantwithminmaxvars_readvariableop_resource: M
Cmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: 
identity¢BiasAdd/ReadVariableOp¢5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp¢7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1¢7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2¢8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp¢:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1¢0dense_1/kernel/Regularizer/L2Loss/ReadVariableOp¶
5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp>lastvaluequant_fakequantwithminmaxvars_readvariableop_resource* 
_output_shapes
:
*
dtype0°
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp@lastvaluequant_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0°
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2ReadVariableOp@lastvaluequant_fakequantwithminmaxvars_readvariableop_2_resource*
_output_shapes
: *
dtype0¸
&LastValueQuant/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars=LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp:value:0?LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0?LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2:value:0* 
_output_shapes
:
*
narrow_range(}
MatMulMatMulinputs0LastValueQuant/FakeQuantWithMinMaxVars:outputs:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ²
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpAmovingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype0¶
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpCmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0
)MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsRelu:activations:0@MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BMovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ±
0dense_1/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp>lastvaluequant_fakequantwithminmaxvars_readvariableop_resource* 
_output_shapes
:
*
dtype0
!dense_1/kernel/Regularizer/L2LossL2Loss8dense_1/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: e
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0*dense_1/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: 
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¶
NoOpNoOp^BiasAdd/ReadVariableOp6^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp8^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_18^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_29^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_11^dense_1/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ: : : : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2n
5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp2r
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_17LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_12r
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_27LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_22t
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12d
0dense_1/kernel/Regularizer/L2Loss/ReadVariableOp0dense_1/kernel/Regularizer/L2Loss/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ç
J
.__inference_quant_flatten_layer_call_fn_146733

inputs
identityÃ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *;
config_proto+)

CPU

GPU2*0J
	
   E 8 *R
fMRK
I__inference_quant_flatten_layer_call_and_return_conditional_losses_143759a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ :W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
¨T
	
G__inference_quant_dense_layer_call_and_return_conditional_losses_146857

inputsC
/lastvaluequant_batchmin_readvariableop_resource:
/
%lastvaluequant_assignminlast_resource: /
%lastvaluequant_assignmaxlast_resource: .
biasadd_readvariableop_resource:	@
6movingavgquantize_assignminema_readvariableop_resource: @
6movingavgquantize_assignmaxema_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢LastValueQuant/AssignMaxLast¢LastValueQuant/AssignMinLast¢&LastValueQuant/BatchMax/ReadVariableOp¢&LastValueQuant/BatchMin/ReadVariableOp¢5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp¢7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1¢7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2¢2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp¢-MovingAvgQuantize/AssignMaxEma/ReadVariableOp¢2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp¢-MovingAvgQuantize/AssignMinEma/ReadVariableOp¢8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp¢:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1¢.dense/kernel/Regularizer/L2Loss/ReadVariableOpe
LastValueQuant/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
&LastValueQuant/BatchMin/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource* 
_output_shapes
:
*
dtype0
LastValueQuant/BatchMinMin.LastValueQuant/BatchMin/ReadVariableOp:value:0LastValueQuant/Const:output:0*
T0*
_output_shapes
: g
LastValueQuant/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       
&LastValueQuant/BatchMax/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource* 
_output_shapes
:
*
dtype0
LastValueQuant/BatchMaxMax.LastValueQuant/BatchMax/ReadVariableOp:value:0LastValueQuant/Const_1:output:0*
T0*
_output_shapes
: ]
LastValueQuant/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ¿
LastValueQuant/truedivRealDiv LastValueQuant/BatchMax:output:0!LastValueQuant/truediv/y:output:0*
T0*
_output_shapes
: 
LastValueQuant/MinimumMinimum LastValueQuant/BatchMin:output:0LastValueQuant/truediv:z:0*
T0*
_output_shapes
: Y
LastValueQuant/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ¿{
LastValueQuant/mulMul LastValueQuant/BatchMin:output:0LastValueQuant/mul/y:output:0*
T0*
_output_shapes
: |
LastValueQuant/MaximumMaximum LastValueQuant/BatchMax:output:0LastValueQuant/mul:z:0*
T0*
_output_shapes
: ¬
LastValueQuant/AssignMinLastAssignVariableOp%lastvaluequant_assignminlast_resourceLastValueQuant/Minimum:z:0*
_output_shapes
 *
dtype0*
validate_shape(¬
LastValueQuant/AssignMaxLastAssignVariableOp%lastvaluequant_assignmaxlast_resourceLastValueQuant/Maximum:z:0*
_output_shapes
 *
dtype0*
validate_shape(§
5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource* 
_output_shapes
:
*
dtype0´
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp%lastvaluequant_assignminlast_resource^LastValueQuant/AssignMinLast*
_output_shapes
: *
dtype0´
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2ReadVariableOp%lastvaluequant_assignmaxlast_resource^LastValueQuant/AssignMaxLast*
_output_shapes
: *
dtype0¸
&LastValueQuant/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars=LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp:value:0?LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0?LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2:value:0* 
_output_shapes
:
*
narrow_range(}
MatMulMatMulinputs0LastValueQuant/FakeQuantWithMinMaxVars:outputs:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
MovingAvgQuantize/ConstConst*
_output_shapes
:*
dtype0*
valueB"       x
MovingAvgQuantize/BatchMinMinRelu:activations:0 MovingAvgQuantize/Const:output:0*
T0*
_output_shapes
: j
MovingAvgQuantize/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       z
MovingAvgQuantize/BatchMaxMaxRelu:activations:0"MovingAvgQuantize/Const_1:output:0*
T0*
_output_shapes
: `
MovingAvgQuantize/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
MovingAvgQuantize/MinimumMinimum#MovingAvgQuantize/BatchMin:output:0$MovingAvgQuantize/Minimum/y:output:0*
T0*
_output_shapes
: `
MovingAvgQuantize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
MovingAvgQuantize/MaximumMaximum#MovingAvgQuantize/BatchMax:output:0$MovingAvgQuantize/Maximum/y:output:0*
T0*
_output_shapes
: i
$MovingAvgQuantize/AssignMinEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o:
-MovingAvgQuantize/AssignMinEma/ReadVariableOpReadVariableOp6movingavgquantize_assignminema_readvariableop_resource*
_output_shapes
: *
dtype0 
"MovingAvgQuantize/AssignMinEma/subSub5MovingAvgQuantize/AssignMinEma/ReadVariableOp:value:0MovingAvgQuantize/Minimum:z:0*
T0*
_output_shapes
: ¡
"MovingAvgQuantize/AssignMinEma/mulMul&MovingAvgQuantize/AssignMinEma/sub:z:0-MovingAvgQuantize/AssignMinEma/decay:output:0*
T0*
_output_shapes
: ü
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOpAssignSubVariableOp6movingavgquantize_assignminema_readvariableop_resource&MovingAvgQuantize/AssignMinEma/mul:z:0.^MovingAvgQuantize/AssignMinEma/ReadVariableOp*
_output_shapes
 *
dtype0i
$MovingAvgQuantize/AssignMaxEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o:
-MovingAvgQuantize/AssignMaxEma/ReadVariableOpReadVariableOp6movingavgquantize_assignmaxema_readvariableop_resource*
_output_shapes
: *
dtype0 
"MovingAvgQuantize/AssignMaxEma/subSub5MovingAvgQuantize/AssignMaxEma/ReadVariableOp:value:0MovingAvgQuantize/Maximum:z:0*
T0*
_output_shapes
: ¡
"MovingAvgQuantize/AssignMaxEma/mulMul&MovingAvgQuantize/AssignMaxEma/sub:z:0-MovingAvgQuantize/AssignMaxEma/decay:output:0*
T0*
_output_shapes
: ü
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpAssignSubVariableOp6movingavgquantize_assignmaxema_readvariableop_resource&MovingAvgQuantize/AssignMaxEma/mul:z:0.^MovingAvgQuantize/AssignMaxEma/ReadVariableOp*
_output_shapes
 *
dtype0Ü
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp6movingavgquantize_assignminema_readvariableop_resource3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp*
_output_shapes
: *
dtype0Þ
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp6movingavgquantize_assignmaxema_readvariableop_resource3^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp*
_output_shapes
: *
dtype0
)MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsRelu:activations:0@MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BMovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
.dense/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource* 
_output_shapes
:
*
dtype0
dense/kernel/Regularizer/L2LossL2Loss6dense/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: c
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0(dense/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: 
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp^BiasAdd/ReadVariableOp^LastValueQuant/AssignMaxLast^LastValueQuant/AssignMinLast'^LastValueQuant/BatchMax/ReadVariableOp'^LastValueQuant/BatchMin/ReadVariableOp6^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp8^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_18^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_23^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp.^MovingAvgQuantize/AssignMaxEma/ReadVariableOp3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp.^MovingAvgQuantize/AssignMinEma/ReadVariableOp9^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1/^dense/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ: : : : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2<
LastValueQuant/AssignMaxLastLastValueQuant/AssignMaxLast2<
LastValueQuant/AssignMinLastLastValueQuant/AssignMinLast2P
&LastValueQuant/BatchMax/ReadVariableOp&LastValueQuant/BatchMax/ReadVariableOp2P
&LastValueQuant/BatchMin/ReadVariableOp&LastValueQuant/BatchMin/ReadVariableOp2n
5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp2r
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_17LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_12r
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_27LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_22h
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2^
-MovingAvgQuantize/AssignMaxEma/ReadVariableOp-MovingAvgQuantize/AssignMaxEma/ReadVariableOp2h
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2^
-MovingAvgQuantize/AssignMinEma/ReadVariableOp-MovingAvgQuantize/AssignMinEma/ReadVariableOp2t
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12`
.dense/kernel/Regularizer/L2Loss/ReadVariableOp.dense/kernel/Regularizer/L2Loss/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


j
K__inference_quant_dropout_1_layer_call_and_return_conditional_losses_143543

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?§
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿp
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
³N
Ø
I__inference_quant_dense_2_layer_call_and_return_conditional_losses_147135

inputsB
/lastvaluequant_batchmin_readvariableop_resource:	/
%lastvaluequant_assignminlast_resource: /
%lastvaluequant_assignmaxlast_resource: -
biasadd_readvariableop_resource:@
6movingavgquantize_assignminema_readvariableop_resource: @
6movingavgquantize_assignmaxema_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢LastValueQuant/AssignMaxLast¢LastValueQuant/AssignMinLast¢&LastValueQuant/BatchMax/ReadVariableOp¢&LastValueQuant/BatchMin/ReadVariableOp¢5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp¢7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1¢7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2¢2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp¢-MovingAvgQuantize/AssignMaxEma/ReadVariableOp¢2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp¢-MovingAvgQuantize/AssignMinEma/ReadVariableOp¢8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp¢:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1e
LastValueQuant/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
&LastValueQuant/BatchMin/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource*
_output_shapes
:	*
dtype0
LastValueQuant/BatchMinMin.LastValueQuant/BatchMin/ReadVariableOp:value:0LastValueQuant/Const:output:0*
T0*
_output_shapes
: g
LastValueQuant/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       
&LastValueQuant/BatchMax/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource*
_output_shapes
:	*
dtype0
LastValueQuant/BatchMaxMax.LastValueQuant/BatchMax/ReadVariableOp:value:0LastValueQuant/Const_1:output:0*
T0*
_output_shapes
: ]
LastValueQuant/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ¿
LastValueQuant/truedivRealDiv LastValueQuant/BatchMax:output:0!LastValueQuant/truediv/y:output:0*
T0*
_output_shapes
: 
LastValueQuant/MinimumMinimum LastValueQuant/BatchMin:output:0LastValueQuant/truediv:z:0*
T0*
_output_shapes
: Y
LastValueQuant/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ¿{
LastValueQuant/mulMul LastValueQuant/BatchMin:output:0LastValueQuant/mul/y:output:0*
T0*
_output_shapes
: |
LastValueQuant/MaximumMaximum LastValueQuant/BatchMax:output:0LastValueQuant/mul:z:0*
T0*
_output_shapes
: ¬
LastValueQuant/AssignMinLastAssignVariableOp%lastvaluequant_assignminlast_resourceLastValueQuant/Minimum:z:0*
_output_shapes
 *
dtype0*
validate_shape(¬
LastValueQuant/AssignMaxLastAssignVariableOp%lastvaluequant_assignmaxlast_resourceLastValueQuant/Maximum:z:0*
_output_shapes
 *
dtype0*
validate_shape(¦
5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource*
_output_shapes
:	*
dtype0´
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp%lastvaluequant_assignminlast_resource^LastValueQuant/AssignMinLast*
_output_shapes
: *
dtype0´
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2ReadVariableOp%lastvaluequant_assignmaxlast_resource^LastValueQuant/AssignMaxLast*
_output_shapes
: *
dtype0·
&LastValueQuant/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars=LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp:value:0?LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0?LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2:value:0*
_output_shapes
:	*
narrow_range(|
MatMulMatMulinputs0LastValueQuant/FakeQuantWithMinMaxVars:outputs:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
MovingAvgQuantize/ConstConst*
_output_shapes
:*
dtype0*
valueB"       v
MovingAvgQuantize/BatchMinMinBiasAdd:output:0 MovingAvgQuantize/Const:output:0*
T0*
_output_shapes
: j
MovingAvgQuantize/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       x
MovingAvgQuantize/BatchMaxMaxBiasAdd:output:0"MovingAvgQuantize/Const_1:output:0*
T0*
_output_shapes
: `
MovingAvgQuantize/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
MovingAvgQuantize/MinimumMinimum#MovingAvgQuantize/BatchMin:output:0$MovingAvgQuantize/Minimum/y:output:0*
T0*
_output_shapes
: `
MovingAvgQuantize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
MovingAvgQuantize/MaximumMaximum#MovingAvgQuantize/BatchMax:output:0$MovingAvgQuantize/Maximum/y:output:0*
T0*
_output_shapes
: i
$MovingAvgQuantize/AssignMinEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o:
-MovingAvgQuantize/AssignMinEma/ReadVariableOpReadVariableOp6movingavgquantize_assignminema_readvariableop_resource*
_output_shapes
: *
dtype0 
"MovingAvgQuantize/AssignMinEma/subSub5MovingAvgQuantize/AssignMinEma/ReadVariableOp:value:0MovingAvgQuantize/Minimum:z:0*
T0*
_output_shapes
: ¡
"MovingAvgQuantize/AssignMinEma/mulMul&MovingAvgQuantize/AssignMinEma/sub:z:0-MovingAvgQuantize/AssignMinEma/decay:output:0*
T0*
_output_shapes
: ü
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOpAssignSubVariableOp6movingavgquantize_assignminema_readvariableop_resource&MovingAvgQuantize/AssignMinEma/mul:z:0.^MovingAvgQuantize/AssignMinEma/ReadVariableOp*
_output_shapes
 *
dtype0i
$MovingAvgQuantize/AssignMaxEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o:
-MovingAvgQuantize/AssignMaxEma/ReadVariableOpReadVariableOp6movingavgquantize_assignmaxema_readvariableop_resource*
_output_shapes
: *
dtype0 
"MovingAvgQuantize/AssignMaxEma/subSub5MovingAvgQuantize/AssignMaxEma/ReadVariableOp:value:0MovingAvgQuantize/Maximum:z:0*
T0*
_output_shapes
: ¡
"MovingAvgQuantize/AssignMaxEma/mulMul&MovingAvgQuantize/AssignMaxEma/sub:z:0-MovingAvgQuantize/AssignMaxEma/decay:output:0*
T0*
_output_shapes
: ü
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpAssignSubVariableOp6movingavgquantize_assignmaxema_readvariableop_resource&MovingAvgQuantize/AssignMaxEma/mul:z:0.^MovingAvgQuantize/AssignMaxEma/ReadVariableOp*
_output_shapes
 *
dtype0Ü
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp6movingavgquantize_assignminema_readvariableop_resource3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp*
_output_shapes
: *
dtype0Þ
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp6movingavgquantize_assignmaxema_readvariableop_resource3^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp*
_output_shapes
: *
dtype0
)MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsBiasAdd:output:0@MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BMovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿy
SoftmaxSoftmax3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÝ
NoOpNoOp^BiasAdd/ReadVariableOp^LastValueQuant/AssignMaxLast^LastValueQuant/AssignMinLast'^LastValueQuant/BatchMax/ReadVariableOp'^LastValueQuant/BatchMin/ReadVariableOp6^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp8^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_18^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_23^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp.^MovingAvgQuantize/AssignMaxEma/ReadVariableOp3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp.^MovingAvgQuantize/AssignMinEma/ReadVariableOp9^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ: : : : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2<
LastValueQuant/AssignMaxLastLastValueQuant/AssignMaxLast2<
LastValueQuant/AssignMinLastLastValueQuant/AssignMinLast2P
&LastValueQuant/BatchMax/ReadVariableOp&LastValueQuant/BatchMax/ReadVariableOp2P
&LastValueQuant/BatchMin/ReadVariableOp&LastValueQuant/BatchMin/ReadVariableOp2n
5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp2r
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_17LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_12r
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_27LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_22h
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2^
-MovingAvgQuantize/AssignMaxEma/ReadVariableOp-MovingAvgQuantize/AssignMaxEma/ReadVariableOp2h
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2^
-MovingAvgQuantize/AssignMinEma/ReadVariableOp-MovingAvgQuantize/AssignMinEma/ReadVariableOp2t
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

g
K__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_142931

inputs
identity¢
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
½
L
0__inference_quant_dropout_1_layer_call_fn_147009

inputs
identityÅ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *;
config_proto+)

CPU

GPU2*0J
	
   E 8 *T
fORM
K__inference_quant_dropout_1_layer_call_and_return_conditional_losses_143270a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
­X
º	
J__inference_quant_conv2d_4_layer_call_and_return_conditional_losses_146699

inputsI
/lastvaluequant_batchmin_readvariableop_resource:@ 3
%lastvaluequant_assignminlast_resource: 3
%lastvaluequant_assignmaxlast_resource: -
biasadd_readvariableop_resource: @
6movingavgquantize_assignminema_readvariableop_resource: @
6movingavgquantize_assignmaxema_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢LastValueQuant/AssignMaxLast¢LastValueQuant/AssignMinLast¢&LastValueQuant/BatchMax/ReadVariableOp¢&LastValueQuant/BatchMin/ReadVariableOp¢?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp¢ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1¢ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2¢2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp¢-MovingAvgQuantize/AssignMaxEma/ReadVariableOp¢2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp¢-MovingAvgQuantize/AssignMinEma/ReadVariableOp¢8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp¢:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1¢1conv2d_4/kernel/Regularizer/L2Loss/ReadVariableOp
&LastValueQuant/BatchMin/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:@ *
dtype0~
)LastValueQuant/BatchMin/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          §
LastValueQuant/BatchMinMin.LastValueQuant/BatchMin/ReadVariableOp:value:02LastValueQuant/BatchMin/reduction_indices:output:0*
T0*
_output_shapes
: 
&LastValueQuant/BatchMax/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:@ *
dtype0~
)LastValueQuant/BatchMax/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          §
LastValueQuant/BatchMaxMax.LastValueQuant/BatchMax/ReadVariableOp:value:02LastValueQuant/BatchMax/reduction_indices:output:0*
T0*
_output_shapes
: ]
LastValueQuant/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ¿
LastValueQuant/truedivRealDiv LastValueQuant/BatchMax:output:0!LastValueQuant/truediv/y:output:0*
T0*
_output_shapes
: 
LastValueQuant/MinimumMinimum LastValueQuant/BatchMin:output:0LastValueQuant/truediv:z:0*
T0*
_output_shapes
: Y
LastValueQuant/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ¿
LastValueQuant/mulMul LastValueQuant/BatchMin:output:0LastValueQuant/mul/y:output:0*
T0*
_output_shapes
: 
LastValueQuant/MaximumMaximum LastValueQuant/BatchMax:output:0LastValueQuant/mul:z:0*
T0*
_output_shapes
: ¬
LastValueQuant/AssignMinLastAssignVariableOp%lastvaluequant_assignminlast_resourceLastValueQuant/Minimum:z:0*
_output_shapes
 *
dtype0*
validate_shape(¬
LastValueQuant/AssignMaxLastAssignVariableOp%lastvaluequant_assignmaxlast_resourceLastValueQuant/Maximum:z:0*
_output_shapes
 *
dtype0*
validate_shape(·
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:@ *
dtype0Â
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ReadVariableOp%lastvaluequant_assignminlast_resource^LastValueQuant/AssignMinLast*
_output_shapes
: *
dtype0Â
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ReadVariableOp%lastvaluequant_assignmaxlast_resource^LastValueQuant/AssignMaxLast*
_output_shapes
: *
dtype0ð
0LastValueQuant/FakeQuantWithMinMaxVarsPerChannel!FakeQuantWithMinMaxVarsPerChannelGLastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2:value:0*&
_output_shapes
:@ *
narrow_range(¶
Conv2DConv2Dinputs:LastValueQuant/FakeQuantWithMinMaxVarsPerChannel:outputs:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ p
MovingAvgQuantize/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             x
MovingAvgQuantize/BatchMinMinRelu:activations:0 MovingAvgQuantize/Const:output:0*
T0*
_output_shapes
: r
MovingAvgQuantize/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             z
MovingAvgQuantize/BatchMaxMaxRelu:activations:0"MovingAvgQuantize/Const_1:output:0*
T0*
_output_shapes
: `
MovingAvgQuantize/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
MovingAvgQuantize/MinimumMinimum#MovingAvgQuantize/BatchMin:output:0$MovingAvgQuantize/Minimum/y:output:0*
T0*
_output_shapes
: `
MovingAvgQuantize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
MovingAvgQuantize/MaximumMaximum#MovingAvgQuantize/BatchMax:output:0$MovingAvgQuantize/Maximum/y:output:0*
T0*
_output_shapes
: i
$MovingAvgQuantize/AssignMinEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o:
-MovingAvgQuantize/AssignMinEma/ReadVariableOpReadVariableOp6movingavgquantize_assignminema_readvariableop_resource*
_output_shapes
: *
dtype0 
"MovingAvgQuantize/AssignMinEma/subSub5MovingAvgQuantize/AssignMinEma/ReadVariableOp:value:0MovingAvgQuantize/Minimum:z:0*
T0*
_output_shapes
: ¡
"MovingAvgQuantize/AssignMinEma/mulMul&MovingAvgQuantize/AssignMinEma/sub:z:0-MovingAvgQuantize/AssignMinEma/decay:output:0*
T0*
_output_shapes
: ü
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOpAssignSubVariableOp6movingavgquantize_assignminema_readvariableop_resource&MovingAvgQuantize/AssignMinEma/mul:z:0.^MovingAvgQuantize/AssignMinEma/ReadVariableOp*
_output_shapes
 *
dtype0i
$MovingAvgQuantize/AssignMaxEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o:
-MovingAvgQuantize/AssignMaxEma/ReadVariableOpReadVariableOp6movingavgquantize_assignmaxema_readvariableop_resource*
_output_shapes
: *
dtype0 
"MovingAvgQuantize/AssignMaxEma/subSub5MovingAvgQuantize/AssignMaxEma/ReadVariableOp:value:0MovingAvgQuantize/Maximum:z:0*
T0*
_output_shapes
: ¡
"MovingAvgQuantize/AssignMaxEma/mulMul&MovingAvgQuantize/AssignMaxEma/sub:z:0-MovingAvgQuantize/AssignMaxEma/decay:output:0*
T0*
_output_shapes
: ü
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpAssignSubVariableOp6movingavgquantize_assignmaxema_readvariableop_resource&MovingAvgQuantize/AssignMaxEma/mul:z:0.^MovingAvgQuantize/AssignMaxEma/ReadVariableOp*
_output_shapes
 *
dtype0Ü
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp6movingavgquantize_assignminema_readvariableop_resource3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp*
_output_shapes
: *
dtype0Þ
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp6movingavgquantize_assignmaxema_readvariableop_resource3^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp*
_output_shapes
: *
dtype0
)MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsRelu:activations:0@MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BMovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ©
1conv2d_4/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:@ *
dtype0
"conv2d_4/kernel/Regularizer/L2LossL2Loss9conv2d_4/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!conv2d_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o; 
conv2d_4/kernel/Regularizer/mulMul*conv2d_4/kernel/Regularizer/mul/x:output:0+conv2d_4/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: 
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¯
NoOpNoOp^BiasAdd/ReadVariableOp^LastValueQuant/AssignMaxLast^LastValueQuant/AssignMinLast'^LastValueQuant/BatchMax/ReadVariableOp'^LastValueQuant/BatchMin/ReadVariableOp@^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpB^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1B^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_23^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp.^MovingAvgQuantize/AssignMaxEma/ReadVariableOp3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp.^MovingAvgQuantize/AssignMinEma/ReadVariableOp9^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12^conv2d_4/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ@: : : : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2<
LastValueQuant/AssignMaxLastLastValueQuant/AssignMaxLast2<
LastValueQuant/AssignMinLastLastValueQuant/AssignMinLast2P
&LastValueQuant/BatchMax/ReadVariableOp&LastValueQuant/BatchMax/ReadVariableOp2P
&LastValueQuant/BatchMin/ReadVariableOp&LastValueQuant/BatchMin/ReadVariableOp2
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp2
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_12
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_22h
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2^
-MovingAvgQuantize/AssignMaxEma/ReadVariableOp-MovingAvgQuantize/AssignMaxEma/ReadVariableOp2h
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2^
-MovingAvgQuantize/AssignMinEma/ReadVariableOp-MovingAvgQuantize/AssignMinEma/ReadVariableOp2t
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12f
1conv2d_4/kernel/Regularizer/L2Loss/ReadVariableOp1conv2d_4/kernel/Regularizer/L2Loss/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
¨T
	
G__inference_quant_dense_layer_call_and_return_conditional_losses_143730

inputsC
/lastvaluequant_batchmin_readvariableop_resource:
/
%lastvaluequant_assignminlast_resource: /
%lastvaluequant_assignmaxlast_resource: .
biasadd_readvariableop_resource:	@
6movingavgquantize_assignminema_readvariableop_resource: @
6movingavgquantize_assignmaxema_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢LastValueQuant/AssignMaxLast¢LastValueQuant/AssignMinLast¢&LastValueQuant/BatchMax/ReadVariableOp¢&LastValueQuant/BatchMin/ReadVariableOp¢5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp¢7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1¢7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2¢2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp¢-MovingAvgQuantize/AssignMaxEma/ReadVariableOp¢2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp¢-MovingAvgQuantize/AssignMinEma/ReadVariableOp¢8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp¢:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1¢.dense/kernel/Regularizer/L2Loss/ReadVariableOpe
LastValueQuant/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
&LastValueQuant/BatchMin/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource* 
_output_shapes
:
*
dtype0
LastValueQuant/BatchMinMin.LastValueQuant/BatchMin/ReadVariableOp:value:0LastValueQuant/Const:output:0*
T0*
_output_shapes
: g
LastValueQuant/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       
&LastValueQuant/BatchMax/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource* 
_output_shapes
:
*
dtype0
LastValueQuant/BatchMaxMax.LastValueQuant/BatchMax/ReadVariableOp:value:0LastValueQuant/Const_1:output:0*
T0*
_output_shapes
: ]
LastValueQuant/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ¿
LastValueQuant/truedivRealDiv LastValueQuant/BatchMax:output:0!LastValueQuant/truediv/y:output:0*
T0*
_output_shapes
: 
LastValueQuant/MinimumMinimum LastValueQuant/BatchMin:output:0LastValueQuant/truediv:z:0*
T0*
_output_shapes
: Y
LastValueQuant/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ¿{
LastValueQuant/mulMul LastValueQuant/BatchMin:output:0LastValueQuant/mul/y:output:0*
T0*
_output_shapes
: |
LastValueQuant/MaximumMaximum LastValueQuant/BatchMax:output:0LastValueQuant/mul:z:0*
T0*
_output_shapes
: ¬
LastValueQuant/AssignMinLastAssignVariableOp%lastvaluequant_assignminlast_resourceLastValueQuant/Minimum:z:0*
_output_shapes
 *
dtype0*
validate_shape(¬
LastValueQuant/AssignMaxLastAssignVariableOp%lastvaluequant_assignmaxlast_resourceLastValueQuant/Maximum:z:0*
_output_shapes
 *
dtype0*
validate_shape(§
5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource* 
_output_shapes
:
*
dtype0´
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp%lastvaluequant_assignminlast_resource^LastValueQuant/AssignMinLast*
_output_shapes
: *
dtype0´
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2ReadVariableOp%lastvaluequant_assignmaxlast_resource^LastValueQuant/AssignMaxLast*
_output_shapes
: *
dtype0¸
&LastValueQuant/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars=LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp:value:0?LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0?LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2:value:0* 
_output_shapes
:
*
narrow_range(}
MatMulMatMulinputs0LastValueQuant/FakeQuantWithMinMaxVars:outputs:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
MovingAvgQuantize/ConstConst*
_output_shapes
:*
dtype0*
valueB"       x
MovingAvgQuantize/BatchMinMinRelu:activations:0 MovingAvgQuantize/Const:output:0*
T0*
_output_shapes
: j
MovingAvgQuantize/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       z
MovingAvgQuantize/BatchMaxMaxRelu:activations:0"MovingAvgQuantize/Const_1:output:0*
T0*
_output_shapes
: `
MovingAvgQuantize/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
MovingAvgQuantize/MinimumMinimum#MovingAvgQuantize/BatchMin:output:0$MovingAvgQuantize/Minimum/y:output:0*
T0*
_output_shapes
: `
MovingAvgQuantize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
MovingAvgQuantize/MaximumMaximum#MovingAvgQuantize/BatchMax:output:0$MovingAvgQuantize/Maximum/y:output:0*
T0*
_output_shapes
: i
$MovingAvgQuantize/AssignMinEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o:
-MovingAvgQuantize/AssignMinEma/ReadVariableOpReadVariableOp6movingavgquantize_assignminema_readvariableop_resource*
_output_shapes
: *
dtype0 
"MovingAvgQuantize/AssignMinEma/subSub5MovingAvgQuantize/AssignMinEma/ReadVariableOp:value:0MovingAvgQuantize/Minimum:z:0*
T0*
_output_shapes
: ¡
"MovingAvgQuantize/AssignMinEma/mulMul&MovingAvgQuantize/AssignMinEma/sub:z:0-MovingAvgQuantize/AssignMinEma/decay:output:0*
T0*
_output_shapes
: ü
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOpAssignSubVariableOp6movingavgquantize_assignminema_readvariableop_resource&MovingAvgQuantize/AssignMinEma/mul:z:0.^MovingAvgQuantize/AssignMinEma/ReadVariableOp*
_output_shapes
 *
dtype0i
$MovingAvgQuantize/AssignMaxEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o:
-MovingAvgQuantize/AssignMaxEma/ReadVariableOpReadVariableOp6movingavgquantize_assignmaxema_readvariableop_resource*
_output_shapes
: *
dtype0 
"MovingAvgQuantize/AssignMaxEma/subSub5MovingAvgQuantize/AssignMaxEma/ReadVariableOp:value:0MovingAvgQuantize/Maximum:z:0*
T0*
_output_shapes
: ¡
"MovingAvgQuantize/AssignMaxEma/mulMul&MovingAvgQuantize/AssignMaxEma/sub:z:0-MovingAvgQuantize/AssignMaxEma/decay:output:0*
T0*
_output_shapes
: ü
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpAssignSubVariableOp6movingavgquantize_assignmaxema_readvariableop_resource&MovingAvgQuantize/AssignMaxEma/mul:z:0.^MovingAvgQuantize/AssignMaxEma/ReadVariableOp*
_output_shapes
 *
dtype0Ü
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp6movingavgquantize_assignminema_readvariableop_resource3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp*
_output_shapes
: *
dtype0Þ
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp6movingavgquantize_assignmaxema_readvariableop_resource3^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp*
_output_shapes
: *
dtype0
)MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsRelu:activations:0@MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BMovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
.dense/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource* 
_output_shapes
:
*
dtype0
dense/kernel/Regularizer/L2LossL2Loss6dense/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: c
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0(dense/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: 
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp^BiasAdd/ReadVariableOp^LastValueQuant/AssignMaxLast^LastValueQuant/AssignMinLast'^LastValueQuant/BatchMax/ReadVariableOp'^LastValueQuant/BatchMin/ReadVariableOp6^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp8^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_18^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_23^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp.^MovingAvgQuantize/AssignMaxEma/ReadVariableOp3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp.^MovingAvgQuantize/AssignMinEma/ReadVariableOp9^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1/^dense/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ: : : : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2<
LastValueQuant/AssignMaxLastLastValueQuant/AssignMaxLast2<
LastValueQuant/AssignMinLastLastValueQuant/AssignMinLast2P
&LastValueQuant/BatchMax/ReadVariableOp&LastValueQuant/BatchMax/ReadVariableOp2P
&LastValueQuant/BatchMin/ReadVariableOp&LastValueQuant/BatchMin/ReadVariableOp2n
5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp2r
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_17LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_12r
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_27LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_22h
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2^
-MovingAvgQuantize/AssignMaxEma/ReadVariableOp-MovingAvgQuantize/AssignMaxEma/ReadVariableOp2h
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2^
-MovingAvgQuantize/AssignMinEma/ReadVariableOp-MovingAvgQuantize/AssignMinEma/ReadVariableOp2t
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12`
.dense/kernel/Regularizer/L2Loss/ReadVariableOp.dense/kernel/Regularizer/L2Loss/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

g
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_142919

inputs
identity¢
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
	

/__inference_quant_conv2d_4_layer_call_fn_146604

inputs!
unknown:@ 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: 
identity¢StatefulPartitionedCall©
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *(
_read_only_resource_inputs

*;
config_proto+)

CPU

GPU2*0J
	
   E 8 *S
fNRL
J__inference_quant_conv2d_4_layer_call_and_return_conditional_losses_143151w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ@: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Æ
L
0__inference_max_pooling2d_2_layer_call_fn_147205

inputs
identityç
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *;
config_proto+)

CPU

GPU2*0J
	
   E 8 *T
fORM
K__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_142931
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
­
m
Q__inference_quant_max_pooling2d_1_layer_call_and_return_conditional_losses_144055

inputs
identity
MaxPoolMaxPoolinputs*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
ksize
*
paddingVALID*
strides
`
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿKK :W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿKK 
 
_user_specified_nameinputs
¶
à

+__inference_sequential_layer_call_fn_143439
conv2d_input
unknown: 
	unknown_0: #
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: 
	unknown_5: 
	unknown_6: #
	unknown_7:  
	unknown_8: 
	unknown_9: 

unknown_10: 

unknown_11: 

unknown_12: $

unknown_13: @

unknown_14:@

unknown_15:@

unknown_16:@

unknown_17: 

unknown_18: $

unknown_19:@@

unknown_20:@

unknown_21:@

unknown_22:@

unknown_23: 

unknown_24: $

unknown_25:@ 

unknown_26: 

unknown_27: 

unknown_28: 

unknown_29: 

unknown_30: 

unknown_31:


unknown_32: 

unknown_33: 

unknown_34:	

unknown_35: 

unknown_36: 

unknown_37:


unknown_38: 

unknown_39: 

unknown_40:	

unknown_41: 

unknown_42: 

unknown_43:	

unknown_44: 

unknown_45: 

unknown_46:

unknown_47: 

unknown_48: 
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallconv2d_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48*>
Tin7
523*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*T
_read_only_resource_inputs6
42	
 !"#$%&'()*+,-./012*;
config_proto+)

CPU

GPU2*0J
	
   E 8 *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_143336o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:ÿÿÿÿÿÿÿÿÿææ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿææ
&
_user_specified_nameconv2d_input
ÿ	
h
I__inference_quant_dropout_layer_call_and_return_conditional_losses_146888

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?§
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿp
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ÿ)
Û
J__inference_quant_conv2d_2_layer_call_and_return_conditional_losses_146414

inputsb
Hlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource: @X
Jlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource:@X
Jlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource:@-
biasadd_readvariableop_resource:@K
Amovingavgquantize_fakequantwithminmaxvars_readvariableop_resource: M
Cmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: 
identity¢BiasAdd/ReadVariableOp¢?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp¢ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1¢ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2¢8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp¢:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1¢1conv2d_2/kernel/Regularizer/L2Loss/ReadVariableOpÐ
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpReadVariableOpHlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource*&
_output_shapes
: @*
dtype0È
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ReadVariableOpJlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource*
_output_shapes
:@*
dtype0È
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ReadVariableOpJlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource*
_output_shapes
:@*
dtype0ð
0LastValueQuant/FakeQuantWithMinMaxVarsPerChannel!FakeQuantWithMinMaxVarsPerChannelGLastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2:value:0*&
_output_shapes
: @*
narrow_range(¶
Conv2DConv2Dinputs:LastValueQuant/FakeQuantWithMinMaxVarsPerChannel:outputs:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@²
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpAmovingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype0¶
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpCmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0
)MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsRelu:activations:0@MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BMovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Â
1conv2d_2/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpHlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource*&
_output_shapes
: @*
dtype0
"conv2d_2/kernel/Regularizer/L2LossL2Loss9conv2d_2/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!conv2d_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o; 
conv2d_2/kernel/Regularizer/mulMul*conv2d_2/kernel/Regularizer/mul/x:output:0+conv2d_2/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: 
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Õ
NoOpNoOp^BiasAdd/ReadVariableOp@^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpB^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1B^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_29^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12^conv2d_2/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ : : : : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp2
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_12
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_22t
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12f
1conv2d_2/kernel/Regularizer/L2Loss/ReadVariableOp1conv2d_2/kernel/Regularizer/L2Loss/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
 	
þ
-__inference_quant_conv2d_layer_call_fn_146100

inputs!
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: 
identity¢StatefulPartitionedCall©
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿââ *(
_read_only_resource_inputs

*;
config_proto+)

CPU

GPU2*0J
	
   E 8 *Q
fLRJ
H__inference_quant_conv2d_layer_call_and_return_conditional_losses_142981y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿââ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):ÿÿÿÿÿÿÿÿÿææ: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿææ
 
_user_specified_nameinputs
Ã 
Û
I__inference_quant_dense_2_layer_call_and_return_conditional_losses_147086

inputsQ
>lastvaluequant_fakequantwithminmaxvars_readvariableop_resource:	J
@lastvaluequant_fakequantwithminmaxvars_readvariableop_1_resource: J
@lastvaluequant_fakequantwithminmaxvars_readvariableop_2_resource: -
biasadd_readvariableop_resource:K
Amovingavgquantize_fakequantwithminmaxvars_readvariableop_resource: M
Cmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: 
identity¢BiasAdd/ReadVariableOp¢5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp¢7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1¢7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2¢8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp¢:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1µ
5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp>lastvaluequant_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
:	*
dtype0°
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp@lastvaluequant_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0°
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2ReadVariableOp@lastvaluequant_fakequantwithminmaxvars_readvariableop_2_resource*
_output_shapes
: *
dtype0·
&LastValueQuant/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars=LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp:value:0?LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0?LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2:value:0*
_output_shapes
:	*
narrow_range(|
MatMulMatMulinputs0LastValueQuant/FakeQuantWithMinMaxVars:outputs:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ²
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpAmovingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype0¶
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpCmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0
)MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsBiasAdd:output:0@MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BMovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿy
SoftmaxSoftmax3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp^BiasAdd/ReadVariableOp6^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp8^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_18^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_29^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ: : : : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2n
5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp2r
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_17LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_12r
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_27LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_22t
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

Ú

+__inference_sequential_layer_call_fn_145425

inputs
unknown: 
	unknown_0: #
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: 
	unknown_5: 
	unknown_6: #
	unknown_7:  
	unknown_8: 
	unknown_9: 

unknown_10: 

unknown_11: 

unknown_12: $

unknown_13: @

unknown_14:@

unknown_15:@

unknown_16:@

unknown_17: 

unknown_18: $

unknown_19:@@

unknown_20:@

unknown_21:@

unknown_22:@

unknown_23: 

unknown_24: $

unknown_25:@ 

unknown_26: 

unknown_27: 

unknown_28: 

unknown_29: 

unknown_30: 

unknown_31:


unknown_32: 

unknown_33: 

unknown_34:	

unknown_35: 

unknown_36: 

unknown_37:


unknown_38: 

unknown_39: 

unknown_40:	

unknown_41: 

unknown_42: 

unknown_43:	

unknown_44: 

unknown_45: 

unknown_46:

unknown_47: 

unknown_48: 
identity¢StatefulPartitionedCallÞ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48*>
Tin7
523*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*2
_read_only_resource_inputs
	!$'*-0*;
config_proto+)

CPU

GPU2*0J
	
   E 8 *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_144544o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:ÿÿÿÿÿÿÿÿÿææ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿææ
 
_user_specified_nameinputs"µ	L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Ä
serving_default°
O
conv2d_input?
serving_default_conv2d_input:0ÿÿÿÿÿÿÿÿÿææA
quant_dense_20
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:Ï
¹
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
layer_with_weights-4
layer-4
layer_with_weights-5
layer-5
layer_with_weights-6
layer-6
layer_with_weights-7
layer-7
	layer_with_weights-8
	layer-8

layer_with_weights-9

layer-9
layer_with_weights-10
layer-10
layer_with_weights-11
layer-11
layer_with_weights-12
layer-12
layer_with_weights-13
layer-13
layer_with_weights-14
layer-14
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures"
_tf_keras_sequential
ý
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
quantize_layer_min
 quantize_layer_max
!quantizer_vars
"optimizer_step"
_tf_keras_layer
Û
#	variables
$trainable_variables
%regularization_losses
&	keras_api
'__call__
*(&call_and_return_all_conditional_losses
	)layer
*optimizer_step
+_weight_vars
,
kernel_min
-
kernel_max
._quantize_activations
/post_activation_min
0post_activation_max
1_output_quantizers"
_tf_keras_layer

2	variables
3trainable_variables
4regularization_losses
5	keras_api
6__call__
*7&call_and_return_all_conditional_losses
	8layer
9optimizer_step
:_weight_vars
;_quantize_activations
<_output_quantizers"
_tf_keras_layer
Û
=	variables
>trainable_variables
?regularization_losses
@	keras_api
A__call__
*B&call_and_return_all_conditional_losses
	Clayer
Doptimizer_step
E_weight_vars
F
kernel_min
G
kernel_max
H_quantize_activations
Ipost_activation_min
Jpost_activation_max
K_output_quantizers"
_tf_keras_layer

L	variables
Mtrainable_variables
Nregularization_losses
O	keras_api
P__call__
*Q&call_and_return_all_conditional_losses
	Rlayer
Soptimizer_step
T_weight_vars
U_quantize_activations
V_output_quantizers"
_tf_keras_layer
Û
W	variables
Xtrainable_variables
Yregularization_losses
Z	keras_api
[__call__
*\&call_and_return_all_conditional_losses
	]layer
^optimizer_step
__weight_vars
`
kernel_min
a
kernel_max
b_quantize_activations
cpost_activation_min
dpost_activation_max
e_output_quantizers"
_tf_keras_layer
Û
f	variables
gtrainable_variables
hregularization_losses
i	keras_api
j__call__
*k&call_and_return_all_conditional_losses
	llayer
moptimizer_step
n_weight_vars
o
kernel_min
p
kernel_max
q_quantize_activations
rpost_activation_min
spost_activation_max
t_output_quantizers"
_tf_keras_layer
ß
u	variables
vtrainable_variables
wregularization_losses
x	keras_api
y__call__
*z&call_and_return_all_conditional_losses
	{layer
|optimizer_step
}_weight_vars
~
kernel_min

kernel_max
_quantize_activations
post_activation_min
post_activation_max
_output_quantizers"
_tf_keras_layer

	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses

layer
optimizer_step
_weight_vars
_quantize_activations
_output_quantizers"
_tf_keras_layer

	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses

layer
optimizer_step
_weight_vars
_quantize_activations
_output_quantizers"
_tf_keras_layer
ê
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses

 layer
¡optimizer_step
¢_weight_vars
£
kernel_min
¤
kernel_max
¥_quantize_activations
¦post_activation_min
§post_activation_max
¨_output_quantizers"
_tf_keras_layer

©	variables
ªtrainable_variables
«regularization_losses
¬	keras_api
­__call__
+®&call_and_return_all_conditional_losses

¯layer
°optimizer_step
±_weight_vars
²_quantize_activations
³_output_quantizers"
_tf_keras_layer
ê
´	variables
µtrainable_variables
¶regularization_losses
·	keras_api
¸__call__
+¹&call_and_return_all_conditional_losses

ºlayer
»optimizer_step
¼_weight_vars
½
kernel_min
¾
kernel_max
¿_quantize_activations
Àpost_activation_min
Ápost_activation_max
Â_output_quantizers"
_tf_keras_layer

Ã	variables
Ätrainable_variables
Åregularization_losses
Æ	keras_api
Ç__call__
+È&call_and_return_all_conditional_losses

Élayer
Êoptimizer_step
Ë_weight_vars
Ì_quantize_activations
Í_output_quantizers"
_tf_keras_layer
è
Î	variables
Ïtrainable_variables
Ðregularization_losses
Ñ	keras_api
Ò__call__
+Ó&call_and_return_all_conditional_losses

Ôlayer
Õoptimizer_step
Ö_weight_vars
×
kernel_min
Ø
kernel_max
Ù_quantize_activations
Úpre_activation_min
Ûpre_activation_max
Ü_output_quantizers"
_tf_keras_layer
Ã
0
 1
"2
Ý3
Þ4
*5
,6
-7
/8
09
910
ß11
à12
D13
F14
G15
I16
J17
S18
á19
â20
^21
`22
a23
c24
d25
ã26
ä27
m28
o29
p30
r31
s32
å33
æ34
|35
~36
37
38
39
40
41
ç42
è43
¡44
£45
¤46
¦47
§48
°49
é50
ê51
»52
½53
¾54
À55
Á56
Ê57
ë58
ì59
Õ60
×61
Ø62
Ú63
Û64"
trackable_list_wrapper
¦
Ý0
Þ1
ß2
à3
á4
â5
ã6
ä7
å8
æ9
ç10
è11
é12
ê13
ë14
ì15"
trackable_list_wrapper
 "
trackable_list_wrapper
Ï
ínon_trainable_variables
îlayers
ïmetrics
 ðlayer_regularization_losses
ñlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
é
òtrace_0
ótrace_1
ôtrace_2
õtrace_32ö
+__inference_sequential_layer_call_fn_143439
+__inference_sequential_layer_call_fn_145320
+__inference_sequential_layer_call_fn_145425
+__inference_sequential_layer_call_fn_144752¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zòtrace_0zótrace_1zôtrace_2zõtrace_3
Õ
ötrace_0
÷trace_1
øtrace_2
ùtrace_32â
F__inference_sequential_layer_call_and_return_conditional_losses_145605
F__inference_sequential_layer_call_and_return_conditional_losses_146035
F__inference_sequential_layer_call_and_return_conditional_losses_144899
F__inference_sequential_layer_call_and_return_conditional_losses_145046¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zötrace_0z÷trace_1zøtrace_2zùtrace_3
ÑBÎ
!__inference__wrapped_model_142898conv2d_input"
²
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¸
	úiter
ûbeta_1
übeta_2

ýdecay
þlearning_rate	Ým×	ÞmØ	ßmÙ	àmÚ	ámÛ	âmÜ	ãmÝ	ämÞ	åmß	æmà	çmá	èmâ	émã	êmä	ëmå	ìmæ	Ývç	Þvè	ßvé	àvê	ávë	âvì	ãví	ävî	åvï	ævð	çvñ	èvò	évó	êvô	ëvõ	ìvö"
	optimizer
-
ÿserving_default"
signature_map
5
0
 1
"2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
Ó
trace_0
trace_12
/__inference_quantize_layer_layer_call_fn_146044
/__inference_quantize_layer_layer_call_fn_146053³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ztrace_0ztrace_1

trace_0
trace_12Î
J__inference_quantize_layer_layer_call_and_return_conditional_losses_146062
J__inference_quantize_layer_layer_call_and_return_conditional_losses_146083³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ztrace_0ztrace_1
):' 2!quantize_layer/quantize_layer_min
):' 2!quantize_layer/quantize_layer_max
:
min_var
 max_var"
trackable_dict_wrapper
%:# 2quantize_layer/optimizer_step
S
Ý0
Þ1
*2
,3
-4
/5
06"
trackable_list_wrapper
0
Ý0
Þ1"
trackable_list_wrapper
(
0"
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
#	variables
$trainable_variables
%regularization_losses
'__call__
*(&call_and_return_all_conditional_losses
&("call_and_return_conditional_losses"
_generic_user_object
Õ
trace_0
trace_12
-__inference_quant_conv2d_layer_call_fn_146100
-__inference_quant_conv2d_layer_call_fn_146117¹
°²¬
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ztrace_0ztrace_1

trace_0
trace_12Ð
H__inference_quant_conv2d_layer_call_and_return_conditional_losses_146142
H__inference_quant_conv2d_layer_call_and_return_conditional_losses_146195¹
°²¬
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ztrace_0ztrace_1
æ
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
Ýkernel
	Þbias
!_jit_compiled_convolution_op"
_tf_keras_layer
#:! 2quant_conv2d/optimizer_step
(
0"
trackable_list_wrapper
#:! 2quant_conv2d/kernel_min
#:! 2quant_conv2d/kernel_max
 "
trackable_list_wrapper
(:& 2 quant_conv2d/post_activation_min
(:& 2 quant_conv2d/post_activation_max
 "
trackable_list_wrapper
'
90"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
2	variables
3trainable_variables
4regularization_losses
6__call__
*7&call_and_return_all_conditional_losses
&7"call_and_return_conditional_losses"
_generic_user_object
ã
 trace_0
¡trace_12¨
4__inference_quant_max_pooling2d_layer_call_fn_146204
4__inference_quant_max_pooling2d_layer_call_fn_146209¹
°²¬
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z trace_0z¡trace_1

¢trace_0
£trace_12Þ
O__inference_quant_max_pooling2d_layer_call_and_return_conditional_losses_146214
O__inference_quant_max_pooling2d_layer_call_and_return_conditional_losses_146219¹
°²¬
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z¢trace_0z£trace_1
«
¤	variables
¥trainable_variables
¦regularization_losses
§	keras_api
¨__call__
+©&call_and_return_all_conditional_losses"
_tf_keras_layer
*:( 2"quant_max_pooling2d/optimizer_step
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
S
ß0
à1
D2
F3
G4
I5
J6"
trackable_list_wrapper
0
ß0
à1"
trackable_list_wrapper
(
ª0"
trackable_list_wrapper
²
«non_trainable_variables
¬layers
­metrics
 ®layer_regularization_losses
¯layer_metrics
=	variables
>trainable_variables
?regularization_losses
A__call__
*B&call_and_return_all_conditional_losses
&B"call_and_return_conditional_losses"
_generic_user_object
Ù
°trace_0
±trace_12
/__inference_quant_conv2d_1_layer_call_fn_146236
/__inference_quant_conv2d_1_layer_call_fn_146253¹
°²¬
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z°trace_0z±trace_1

²trace_0
³trace_12Ô
J__inference_quant_conv2d_1_layer_call_and_return_conditional_losses_146278
J__inference_quant_conv2d_1_layer_call_and_return_conditional_losses_146331¹
°²¬
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z²trace_0z³trace_1
æ
´	variables
µtrainable_variables
¶regularization_losses
·	keras_api
¸__call__
+¹&call_and_return_all_conditional_losses
ßkernel
	àbias
!º_jit_compiled_convolution_op"
_tf_keras_layer
%:# 2quant_conv2d_1/optimizer_step
(
»0"
trackable_list_wrapper
%:# 2quant_conv2d_1/kernel_min
%:# 2quant_conv2d_1/kernel_max
 "
trackable_list_wrapper
*:( 2"quant_conv2d_1/post_activation_min
*:( 2"quant_conv2d_1/post_activation_max
 "
trackable_list_wrapper
'
S0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
¼non_trainable_variables
½layers
¾metrics
 ¿layer_regularization_losses
Àlayer_metrics
L	variables
Mtrainable_variables
Nregularization_losses
P__call__
*Q&call_and_return_all_conditional_losses
&Q"call_and_return_conditional_losses"
_generic_user_object
ç
Átrace_0
Âtrace_12¬
6__inference_quant_max_pooling2d_1_layer_call_fn_146340
6__inference_quant_max_pooling2d_1_layer_call_fn_146345¹
°²¬
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zÁtrace_0zÂtrace_1

Ãtrace_0
Ätrace_12â
Q__inference_quant_max_pooling2d_1_layer_call_and_return_conditional_losses_146350
Q__inference_quant_max_pooling2d_1_layer_call_and_return_conditional_losses_146355¹
°²¬
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zÃtrace_0zÄtrace_1
«
Å	variables
Ætrainable_variables
Çregularization_losses
È	keras_api
É__call__
+Ê&call_and_return_all_conditional_losses"
_tf_keras_layer
,:* 2$quant_max_pooling2d_1/optimizer_step
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
S
á0
â1
^2
`3
a4
c5
d6"
trackable_list_wrapper
0
á0
â1"
trackable_list_wrapper
(
Ë0"
trackable_list_wrapper
²
Ìnon_trainable_variables
Ílayers
Îmetrics
 Ïlayer_regularization_losses
Ðlayer_metrics
W	variables
Xtrainable_variables
Yregularization_losses
[__call__
*\&call_and_return_all_conditional_losses
&\"call_and_return_conditional_losses"
_generic_user_object
Ù
Ñtrace_0
Òtrace_12
/__inference_quant_conv2d_2_layer_call_fn_146372
/__inference_quant_conv2d_2_layer_call_fn_146389¹
°²¬
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zÑtrace_0zÒtrace_1

Ótrace_0
Ôtrace_12Ô
J__inference_quant_conv2d_2_layer_call_and_return_conditional_losses_146414
J__inference_quant_conv2d_2_layer_call_and_return_conditional_losses_146467¹
°²¬
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zÓtrace_0zÔtrace_1
æ
Õ	variables
Ötrainable_variables
×regularization_losses
Ø	keras_api
Ù__call__
+Ú&call_and_return_all_conditional_losses
ákernel
	âbias
!Û_jit_compiled_convolution_op"
_tf_keras_layer
%:# 2quant_conv2d_2/optimizer_step
(
Ü0"
trackable_list_wrapper
%:#@2quant_conv2d_2/kernel_min
%:#@2quant_conv2d_2/kernel_max
 "
trackable_list_wrapper
*:( 2"quant_conv2d_2/post_activation_min
*:( 2"quant_conv2d_2/post_activation_max
 "
trackable_list_wrapper
S
ã0
ä1
m2
o3
p4
r5
s6"
trackable_list_wrapper
0
ã0
ä1"
trackable_list_wrapper
(
Ý0"
trackable_list_wrapper
²
Þnon_trainable_variables
ßlayers
àmetrics
 álayer_regularization_losses
âlayer_metrics
f	variables
gtrainable_variables
hregularization_losses
j__call__
*k&call_and_return_all_conditional_losses
&k"call_and_return_conditional_losses"
_generic_user_object
Ù
ãtrace_0
ätrace_12
/__inference_quant_conv2d_3_layer_call_fn_146488
/__inference_quant_conv2d_3_layer_call_fn_146505¹
°²¬
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zãtrace_0zätrace_1

åtrace_0
ætrace_12Ô
J__inference_quant_conv2d_3_layer_call_and_return_conditional_losses_146530
J__inference_quant_conv2d_3_layer_call_and_return_conditional_losses_146583¹
°²¬
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zåtrace_0zætrace_1
æ
ç	variables
ètrainable_variables
éregularization_losses
ê	keras_api
ë__call__
+ì&call_and_return_all_conditional_losses
ãkernel
	äbias
!í_jit_compiled_convolution_op"
_tf_keras_layer
%:# 2quant_conv2d_3/optimizer_step
(
î0"
trackable_list_wrapper
%:#@2quant_conv2d_3/kernel_min
%:#@2quant_conv2d_3/kernel_max
 "
trackable_list_wrapper
*:( 2"quant_conv2d_3/post_activation_min
*:( 2"quant_conv2d_3/post_activation_max
 "
trackable_list_wrapper
U
å0
æ1
|2
~3
4
5
6"
trackable_list_wrapper
0
å0
æ1"
trackable_list_wrapper
(
ï0"
trackable_list_wrapper
²
ðnon_trainable_variables
ñlayers
òmetrics
 ólayer_regularization_losses
ôlayer_metrics
u	variables
vtrainable_variables
wregularization_losses
y__call__
*z&call_and_return_all_conditional_losses
&z"call_and_return_conditional_losses"
_generic_user_object
Ù
õtrace_0
ötrace_12
/__inference_quant_conv2d_4_layer_call_fn_146604
/__inference_quant_conv2d_4_layer_call_fn_146621¹
°²¬
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zõtrace_0zötrace_1

÷trace_0
øtrace_12Ô
J__inference_quant_conv2d_4_layer_call_and_return_conditional_losses_146646
J__inference_quant_conv2d_4_layer_call_and_return_conditional_losses_146699¹
°²¬
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z÷trace_0zøtrace_1
æ
ù	variables
útrainable_variables
ûregularization_losses
ü	keras_api
ý__call__
+þ&call_and_return_all_conditional_losses
åkernel
	æbias
!ÿ_jit_compiled_convolution_op"
_tf_keras_layer
%:# 2quant_conv2d_4/optimizer_step
(
0"
trackable_list_wrapper
%:# 2quant_conv2d_4/kernel_min
%:# 2quant_conv2d_4/kernel_max
 "
trackable_list_wrapper
*:( 2"quant_conv2d_4/post_activation_min
*:( 2"quant_conv2d_4/post_activation_max
 "
trackable_list_wrapper
(
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
ç
trace_0
trace_12¬
6__inference_quant_max_pooling2d_2_layer_call_fn_146708
6__inference_quant_max_pooling2d_2_layer_call_fn_146713¹
°²¬
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ztrace_0ztrace_1

trace_0
trace_12â
Q__inference_quant_max_pooling2d_2_layer_call_and_return_conditional_losses_146718
Q__inference_quant_max_pooling2d_2_layer_call_and_return_conditional_losses_146723¹
°²¬
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ztrace_0ztrace_1
«
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
,:* 2$quant_max_pooling2d_2/optimizer_step
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
×
trace_0
trace_12
.__inference_quant_flatten_layer_call_fn_146728
.__inference_quant_flatten_layer_call_fn_146733¹
°²¬
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ztrace_0ztrace_1

trace_0
trace_12Ò
I__inference_quant_flatten_layer_call_and_return_conditional_losses_146739
I__inference_quant_flatten_layer_call_and_return_conditional_losses_146745¹
°²¬
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ztrace_0ztrace_1
«
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
$:" 2quant_flatten/optimizer_step
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
X
ç0
è1
¡2
£3
¤4
¦5
§6"
trackable_list_wrapper
0
ç0
è1"
trackable_list_wrapper
(
0"
trackable_list_wrapper
¸
 non_trainable_variables
¡layers
¢metrics
 £layer_regularization_losses
¤layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
Ó
¥trace_0
¦trace_12
,__inference_quant_dense_layer_call_fn_146762
,__inference_quant_dense_layer_call_fn_146779¹
°²¬
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z¥trace_0z¦trace_1

§trace_0
¨trace_12Î
G__inference_quant_dense_layer_call_and_return_conditional_losses_146804
G__inference_quant_dense_layer_call_and_return_conditional_losses_146857¹
°²¬
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z§trace_0z¨trace_1
Ã
©	variables
ªtrainable_variables
«regularization_losses
¬	keras_api
­__call__
+®&call_and_return_all_conditional_losses
çkernel
	èbias"
_tf_keras_layer
":  2quant_dense/optimizer_step
(
¯0"
trackable_list_wrapper
: 2quant_dense/kernel_min
: 2quant_dense/kernel_max
 "
trackable_list_wrapper
':% 2quant_dense/post_activation_min
':% 2quant_dense/post_activation_max
 "
trackable_list_wrapper
(
°0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
°non_trainable_variables
±layers
²metrics
 ³layer_regularization_losses
´layer_metrics
©	variables
ªtrainable_variables
«regularization_losses
­__call__
+®&call_and_return_all_conditional_losses
'®"call_and_return_conditional_losses"
_generic_user_object
×
µtrace_0
¶trace_12
.__inference_quant_dropout_layer_call_fn_146866
.__inference_quant_dropout_layer_call_fn_146871¹
°²¬
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zµtrace_0z¶trace_1

·trace_0
¸trace_12Ò
I__inference_quant_dropout_layer_call_and_return_conditional_losses_146876
I__inference_quant_dropout_layer_call_and_return_conditional_losses_146888¹
°²¬
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z·trace_0z¸trace_1
Ã
¹	variables
ºtrainable_variables
»regularization_losses
¼	keras_api
½__call__
+¾&call_and_return_all_conditional_losses
¿_random_generator"
_tf_keras_layer
$:" 2quant_dropout/optimizer_step
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
X
é0
ê1
»2
½3
¾4
À5
Á6"
trackable_list_wrapper
0
é0
ê1"
trackable_list_wrapper
(
À0"
trackable_list_wrapper
¸
Ánon_trainable_variables
Âlayers
Ãmetrics
 Älayer_regularization_losses
Ålayer_metrics
´	variables
µtrainable_variables
¶regularization_losses
¸__call__
+¹&call_and_return_all_conditional_losses
'¹"call_and_return_conditional_losses"
_generic_user_object
×
Ætrace_0
Çtrace_12
.__inference_quant_dense_1_layer_call_fn_146905
.__inference_quant_dense_1_layer_call_fn_146922¹
°²¬
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zÆtrace_0zÇtrace_1

Ètrace_0
Étrace_12Ò
I__inference_quant_dense_1_layer_call_and_return_conditional_losses_146947
I__inference_quant_dense_1_layer_call_and_return_conditional_losses_147000¹
°²¬
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zÈtrace_0zÉtrace_1
Ã
Ê	variables
Ëtrainable_variables
Ìregularization_losses
Í	keras_api
Î__call__
+Ï&call_and_return_all_conditional_losses
ékernel
	êbias"
_tf_keras_layer
$:" 2quant_dense_1/optimizer_step
(
Ð0"
trackable_list_wrapper
 : 2quant_dense_1/kernel_min
 : 2quant_dense_1/kernel_max
 "
trackable_list_wrapper
):' 2!quant_dense_1/post_activation_min
):' 2!quant_dense_1/post_activation_max
 "
trackable_list_wrapper
(
Ê0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Ñnon_trainable_variables
Òlayers
Ómetrics
 Ôlayer_regularization_losses
Õlayer_metrics
Ã	variables
Ätrainable_variables
Åregularization_losses
Ç__call__
+È&call_and_return_all_conditional_losses
'È"call_and_return_conditional_losses"
_generic_user_object
Û
Ötrace_0
×trace_12 
0__inference_quant_dropout_1_layer_call_fn_147009
0__inference_quant_dropout_1_layer_call_fn_147014¹
°²¬
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zÖtrace_0z×trace_1

Øtrace_0
Ùtrace_12Ö
K__inference_quant_dropout_1_layer_call_and_return_conditional_losses_147019
K__inference_quant_dropout_1_layer_call_and_return_conditional_losses_147031¹
°²¬
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zØtrace_0zÙtrace_1
Ã
Ú	variables
Ûtrainable_variables
Üregularization_losses
Ý	keras_api
Þ__call__
+ß&call_and_return_all_conditional_losses
à_random_generator"
_tf_keras_layer
&:$ 2quant_dropout_1/optimizer_step
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
X
ë0
ì1
Õ2
×3
Ø4
Ú5
Û6"
trackable_list_wrapper
0
ë0
ì1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
ánon_trainable_variables
âlayers
ãmetrics
 älayer_regularization_losses
ålayer_metrics
Î	variables
Ïtrainable_variables
Ðregularization_losses
Ò__call__
+Ó&call_and_return_all_conditional_losses
'Ó"call_and_return_conditional_losses"
_generic_user_object
×
ætrace_0
çtrace_12
.__inference_quant_dense_2_layer_call_fn_147048
.__inference_quant_dense_2_layer_call_fn_147065¹
°²¬
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zætrace_0zçtrace_1

ètrace_0
étrace_12Ò
I__inference_quant_dense_2_layer_call_and_return_conditional_losses_147086
I__inference_quant_dense_2_layer_call_and_return_conditional_losses_147135¹
°²¬
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zètrace_0zétrace_1
Ã
ê	variables
ëtrainable_variables
ìregularization_losses
í	keras_api
î__call__
+ï&call_and_return_all_conditional_losses
ëkernel
	ìbias"
_tf_keras_layer
$:" 2quant_dense_2/optimizer_step
(
ð0"
trackable_list_wrapper
 : 2quant_dense_2/kernel_min
 : 2quant_dense_2/kernel_max
 "
trackable_list_wrapper
(:& 2 quant_dense_2/pre_activation_min
(:& 2 quant_dense_2/pre_activation_max
 "
trackable_list_wrapper
':% 2conv2d/kernel
: 2conv2d/bias
):'  2conv2d_1/kernel
: 2conv2d_1/bias
):' @2conv2d_2/kernel
:@2conv2d_2/bias
):'@@2conv2d_3/kernel
:@2conv2d_3/bias
):'@ 2conv2d_4/kernel
: 2conv2d_4/bias
 :
2dense/kernel
:2
dense/bias
": 
2dense_1/kernel
:2dense_1/bias
!:	2dense_2/kernel
:2dense_2/bias
³
0
 1
"2
*3
,4
-5
/6
07
98
D9
F10
G11
I12
J13
S14
^15
`16
a17
c18
d19
m20
o21
p22
r23
s24
|25
~26
27
28
29
30
31
¡32
£33
¤34
¦35
§36
°37
»38
½39
¾40
À41
Á42
Ê43
Õ44
×45
Ø46
Ú47
Û48"
trackable_list_wrapper

0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14"
trackable_list_wrapper
0
ñ0
ò1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
Bÿ
+__inference_sequential_layer_call_fn_143439conv2d_input"¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
üBù
+__inference_sequential_layer_call_fn_145320inputs"¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
üBù
+__inference_sequential_layer_call_fn_145425inputs"¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Bÿ
+__inference_sequential_layer_call_fn_144752conv2d_input"¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
F__inference_sequential_layer_call_and_return_conditional_losses_145605inputs"¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
F__inference_sequential_layer_call_and_return_conditional_losses_146035inputs"¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
F__inference_sequential_layer_call_and_return_conditional_losses_144899conv2d_input"¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
F__inference_sequential_layer_call_and_return_conditional_losses_145046conv2d_input"¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
ÐBÍ
$__inference_signature_wrapper_145187conv2d_input"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
5
0
 1
"2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ôBñ
/__inference_quantize_layer_layer_call_fn_146044inputs"³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ôBñ
/__inference_quantize_layer_layer_call_fn_146053inputs"³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
J__inference_quantize_layer_layer_call_and_return_conditional_losses_146062inputs"³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
J__inference_quantize_layer_layer_call_and_return_conditional_losses_146083inputs"³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ï
ótrace_02°
__inference_loss_fn_0_147144
²
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *¢ zótrace_0
C
*0
,1
-2
/3
04"
trackable_list_wrapper
'
)0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
øBõ
-__inference_quant_conv2d_layer_call_fn_146100inputs"¹
°²¬
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
øBõ
-__inference_quant_conv2d_layer_call_fn_146117inputs"¹
°²¬
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
H__inference_quant_conv2d_layer_call_and_return_conditional_losses_146142inputs"¹
°²¬
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
H__inference_quant_conv2d_layer_call_and_return_conditional_losses_146195inputs"¹
°²¬
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
(
Þ0"
trackable_list_wrapper
(
Þ0"
trackable_list_wrapper
(
0"
trackable_list_wrapper
¸
ônon_trainable_variables
õlayers
ömetrics
 ÷layer_regularization_losses
ølayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
´2±®
£²
FullArgSpec'
args
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 0
1
Ý0
ù2"
trackable_tuple_wrapper
'
90"
trackable_list_wrapper
'
80"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ÿBü
4__inference_quant_max_pooling2d_layer_call_fn_146204inputs"¹
°²¬
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ÿBü
4__inference_quant_max_pooling2d_layer_call_fn_146209inputs"¹
°²¬
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
O__inference_quant_max_pooling2d_layer_call_and_return_conditional_losses_146214inputs"¹
°²¬
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
O__inference_quant_max_pooling2d_layer_call_and_return_conditional_losses_146219inputs"¹
°²¬
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
únon_trainable_variables
ûlayers
ümetrics
 ýlayer_regularization_losses
þlayer_metrics
¤	variables
¥trainable_variables
¦regularization_losses
¨__call__
+©&call_and_return_all_conditional_losses
'©"call_and_return_conditional_losses"
_generic_user_object
ô
ÿtrace_02Õ
.__inference_max_pooling2d_layer_call_fn_147149¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zÿtrace_0

trace_02ð
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_147154¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ztrace_0
Ï
trace_02°
__inference_loss_fn_1_147163
²
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *¢ ztrace_0
C
D0
F1
G2
I3
J4"
trackable_list_wrapper
'
C0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
úB÷
/__inference_quant_conv2d_1_layer_call_fn_146236inputs"¹
°²¬
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
úB÷
/__inference_quant_conv2d_1_layer_call_fn_146253inputs"¹
°²¬
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
J__inference_quant_conv2d_1_layer_call_and_return_conditional_losses_146278inputs"¹
°²¬
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
J__inference_quant_conv2d_1_layer_call_and_return_conditional_losses_146331inputs"¹
°²¬
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
(
à0"
trackable_list_wrapper
(
à0"
trackable_list_wrapper
(
ª0"
trackable_list_wrapper
¸
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
´	variables
µtrainable_variables
¶regularization_losses
¸__call__
+¹&call_and_return_all_conditional_losses
'¹"call_and_return_conditional_losses"
_generic_user_object
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
´2±®
£²
FullArgSpec'
args
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 0
1
ß0
2"
trackable_tuple_wrapper
'
S0"
trackable_list_wrapper
'
R0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
Bþ
6__inference_quant_max_pooling2d_1_layer_call_fn_146340inputs"¹
°²¬
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Bþ
6__inference_quant_max_pooling2d_1_layer_call_fn_146345inputs"¹
°²¬
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
Q__inference_quant_max_pooling2d_1_layer_call_and_return_conditional_losses_146350inputs"¹
°²¬
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
Q__inference_quant_max_pooling2d_1_layer_call_and_return_conditional_losses_146355inputs"¹
°²¬
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
Å	variables
Ætrainable_variables
Çregularization_losses
É__call__
+Ê&call_and_return_all_conditional_losses
'Ê"call_and_return_conditional_losses"
_generic_user_object
ö
trace_02×
0__inference_max_pooling2d_1_layer_call_fn_147168¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ztrace_0

trace_02ò
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_147173¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ztrace_0
Ï
trace_02°
__inference_loss_fn_2_147182
²
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *¢ ztrace_0
C
^0
`1
a2
c3
d4"
trackable_list_wrapper
'
]0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
úB÷
/__inference_quant_conv2d_2_layer_call_fn_146372inputs"¹
°²¬
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
úB÷
/__inference_quant_conv2d_2_layer_call_fn_146389inputs"¹
°²¬
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
J__inference_quant_conv2d_2_layer_call_and_return_conditional_losses_146414inputs"¹
°²¬
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
J__inference_quant_conv2d_2_layer_call_and_return_conditional_losses_146467inputs"¹
°²¬
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
(
â0"
trackable_list_wrapper
(
â0"
trackable_list_wrapper
(
Ë0"
trackable_list_wrapper
¸
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
Õ	variables
Ötrainable_variables
×regularization_losses
Ù__call__
+Ú&call_and_return_all_conditional_losses
'Ú"call_and_return_conditional_losses"
_generic_user_object
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
´2±®
£²
FullArgSpec'
args
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 0
1
á0
2"
trackable_tuple_wrapper
Ï
trace_02°
__inference_loss_fn_3_147191
²
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *¢ ztrace_0
C
m0
o1
p2
r3
s4"
trackable_list_wrapper
'
l0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
úB÷
/__inference_quant_conv2d_3_layer_call_fn_146488inputs"¹
°²¬
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
úB÷
/__inference_quant_conv2d_3_layer_call_fn_146505inputs"¹
°²¬
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
J__inference_quant_conv2d_3_layer_call_and_return_conditional_losses_146530inputs"¹
°²¬
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
J__inference_quant_conv2d_3_layer_call_and_return_conditional_losses_146583inputs"¹
°²¬
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
(
ä0"
trackable_list_wrapper
(
ä0"
trackable_list_wrapper
(
Ý0"
trackable_list_wrapper
¸
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
ç	variables
ètrainable_variables
éregularization_losses
ë__call__
+ì&call_and_return_all_conditional_losses
'ì"call_and_return_conditional_losses"
_generic_user_object
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
´2±®
£²
FullArgSpec'
args
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 0
1
ã0
2"
trackable_tuple_wrapper
Ï
trace_02°
__inference_loss_fn_4_147200
²
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *¢ ztrace_0
E
|0
~1
2
3
4"
trackable_list_wrapper
'
{0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
úB÷
/__inference_quant_conv2d_4_layer_call_fn_146604inputs"¹
°²¬
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
úB÷
/__inference_quant_conv2d_4_layer_call_fn_146621inputs"¹
°²¬
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
J__inference_quant_conv2d_4_layer_call_and_return_conditional_losses_146646inputs"¹
°²¬
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
J__inference_quant_conv2d_4_layer_call_and_return_conditional_losses_146699inputs"¹
°²¬
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
(
æ0"
trackable_list_wrapper
(
æ0"
trackable_list_wrapper
(
ï0"
trackable_list_wrapper
¸
non_trainable_variables
layers
 metrics
 ¡layer_regularization_losses
¢layer_metrics
ù	variables
útrainable_variables
ûregularization_losses
ý__call__
+þ&call_and_return_all_conditional_losses
'þ"call_and_return_conditional_losses"
_generic_user_object
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
´2±®
£²
FullArgSpec'
args
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 0
1
å0
£2"
trackable_tuple_wrapper
(
0"
trackable_list_wrapper
(
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
Bþ
6__inference_quant_max_pooling2d_2_layer_call_fn_146708inputs"¹
°²¬
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Bþ
6__inference_quant_max_pooling2d_2_layer_call_fn_146713inputs"¹
°²¬
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
Q__inference_quant_max_pooling2d_2_layer_call_and_return_conditional_losses_146718inputs"¹
°²¬
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
Q__inference_quant_max_pooling2d_2_layer_call_and_return_conditional_losses_146723inputs"¹
°²¬
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
¤non_trainable_variables
¥layers
¦metrics
 §layer_regularization_losses
¨layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
ö
©trace_02×
0__inference_max_pooling2d_2_layer_call_fn_147205¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z©trace_0

ªtrace_02ò
K__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_147210¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zªtrace_0
(
0"
trackable_list_wrapper
(
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ùBö
.__inference_quant_flatten_layer_call_fn_146728inputs"¹
°²¬
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ùBö
.__inference_quant_flatten_layer_call_fn_146733inputs"¹
°²¬
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
I__inference_quant_flatten_layer_call_and_return_conditional_losses_146739inputs"¹
°²¬
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
I__inference_quant_flatten_layer_call_and_return_conditional_losses_146745inputs"¹
°²¬
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
«non_trainable_variables
¬layers
­metrics
 ®layer_regularization_losses
¯layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ï
°trace_02°
__inference_loss_fn_5_147219
²
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *¢ z°trace_0
H
¡0
£1
¤2
¦3
§4"
trackable_list_wrapper
(
 0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
÷Bô
,__inference_quant_dense_layer_call_fn_146762inputs"¹
°²¬
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
÷Bô
,__inference_quant_dense_layer_call_fn_146779inputs"¹
°²¬
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
G__inference_quant_dense_layer_call_and_return_conditional_losses_146804inputs"¹
°²¬
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
G__inference_quant_dense_layer_call_and_return_conditional_losses_146857inputs"¹
°²¬
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
(
è0"
trackable_list_wrapper
(
è0"
trackable_list_wrapper
(
0"
trackable_list_wrapper
¸
±non_trainable_variables
²layers
³metrics
 ´layer_regularization_losses
µlayer_metrics
©	variables
ªtrainable_variables
«regularization_losses
­__call__
+®&call_and_return_all_conditional_losses
'®"call_and_return_conditional_losses"
_generic_user_object
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
1
ç0
¶2"
trackable_tuple_wrapper
(
°0"
trackable_list_wrapper
(
¯0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ùBö
.__inference_quant_dropout_layer_call_fn_146866inputs"¹
°²¬
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ùBö
.__inference_quant_dropout_layer_call_fn_146871inputs"¹
°²¬
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
I__inference_quant_dropout_layer_call_and_return_conditional_losses_146876inputs"¹
°²¬
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
I__inference_quant_dropout_layer_call_and_return_conditional_losses_146888inputs"¹
°²¬
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
·non_trainable_variables
¸layers
¹metrics
 ºlayer_regularization_losses
»layer_metrics
¹	variables
ºtrainable_variables
»regularization_losses
½__call__
+¾&call_and_return_all_conditional_losses
'¾"call_and_return_conditional_losses"
_generic_user_object
¹2¶³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¹2¶³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
"
_generic_user_object
Ï
¼trace_02°
__inference_loss_fn_6_147228
²
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *¢ z¼trace_0
H
»0
½1
¾2
À3
Á4"
trackable_list_wrapper
(
º0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ùBö
.__inference_quant_dense_1_layer_call_fn_146905inputs"¹
°²¬
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ùBö
.__inference_quant_dense_1_layer_call_fn_146922inputs"¹
°²¬
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
I__inference_quant_dense_1_layer_call_and_return_conditional_losses_146947inputs"¹
°²¬
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
I__inference_quant_dense_1_layer_call_and_return_conditional_losses_147000inputs"¹
°²¬
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
(
ê0"
trackable_list_wrapper
(
ê0"
trackable_list_wrapper
(
À0"
trackable_list_wrapper
¸
½non_trainable_variables
¾layers
¿metrics
 Àlayer_regularization_losses
Álayer_metrics
Ê	variables
Ëtrainable_variables
Ìregularization_losses
Î__call__
+Ï&call_and_return_all_conditional_losses
'Ï"call_and_return_conditional_losses"
_generic_user_object
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
1
é0
Â2"
trackable_tuple_wrapper
(
Ê0"
trackable_list_wrapper
(
É0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ûBø
0__inference_quant_dropout_1_layer_call_fn_147009inputs"¹
°²¬
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ûBø
0__inference_quant_dropout_1_layer_call_fn_147014inputs"¹
°²¬
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
K__inference_quant_dropout_1_layer_call_and_return_conditional_losses_147019inputs"¹
°²¬
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
K__inference_quant_dropout_1_layer_call_and_return_conditional_losses_147031inputs"¹
°²¬
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Ãnon_trainable_variables
Älayers
Åmetrics
 Ælayer_regularization_losses
Çlayer_metrics
Ú	variables
Ûtrainable_variables
Üregularization_losses
Þ__call__
+ß&call_and_return_all_conditional_losses
'ß"call_and_return_conditional_losses"
_generic_user_object
¹2¶³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¹2¶³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
"
_generic_user_object
H
Õ0
×1
Ø2
Ú3
Û4"
trackable_list_wrapper
(
Ô0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ùBö
.__inference_quant_dense_2_layer_call_fn_147048inputs"¹
°²¬
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ùBö
.__inference_quant_dense_2_layer_call_fn_147065inputs"¹
°²¬
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
I__inference_quant_dense_2_layer_call_and_return_conditional_losses_147086inputs"¹
°²¬
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
I__inference_quant_dense_2_layer_call_and_return_conditional_losses_147135inputs"¹
°²¬
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
(
ì0"
trackable_list_wrapper
(
ì0"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Ènon_trainable_variables
Élayers
Êmetrics
 Ëlayer_regularization_losses
Ìlayer_metrics
ê	variables
ëtrainable_variables
ìregularization_losses
î__call__
+ï&call_and_return_all_conditional_losses
'ï"call_and_return_conditional_losses"
_generic_user_object
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
1
ë0
Í2"
trackable_tuple_wrapper
R
Î	variables
Ï	keras_api

Ðtotal

Ñcount"
_tf_keras_metric
c
Ò	variables
Ó	keras_api

Ôtotal

Õcount
Ö
_fn_kwargs"
_tf_keras_metric
³B°
__inference_loss_fn_0_147144"
²
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *¢ 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
0"
trackable_list_wrapper
 "
trackable_dict_wrapper
:
,min_var
-max_var"
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
âBß
.__inference_max_pooling2d_layer_call_fn_147149inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ýBú
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_147154inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
³B°
__inference_loss_fn_1_147163"
²
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *¢ 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
ª0"
trackable_list_wrapper
 "
trackable_dict_wrapper
:
Fmin_var
Gmax_var"
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
äBá
0__inference_max_pooling2d_1_layer_call_fn_147168inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ÿBü
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_147173inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
³B°
__inference_loss_fn_2_147182"
²
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *¢ 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
Ë0"
trackable_list_wrapper
 "
trackable_dict_wrapper
:
`min_var
amax_var"
trackable_dict_wrapper
³B°
__inference_loss_fn_3_147191"
²
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *¢ 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
Ý0"
trackable_list_wrapper
 "
trackable_dict_wrapper
:
omin_var
pmax_var"
trackable_dict_wrapper
³B°
__inference_loss_fn_4_147200"
²
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *¢ 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
ï0"
trackable_list_wrapper
 "
trackable_dict_wrapper
:
~min_var
max_var"
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
äBá
0__inference_max_pooling2d_2_layer_call_fn_147205inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ÿBü
K__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_147210inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
³B°
__inference_loss_fn_5_147219"
²
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *¢ 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
0"
trackable_list_wrapper
 "
trackable_dict_wrapper
<
£min_var
¤max_var"
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
³B°
__inference_loss_fn_6_147228"
²
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *¢ 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
À0"
trackable_list_wrapper
 "
trackable_dict_wrapper
<
½min_var
¾max_var"
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
<
×min_var
Ømax_var"
trackable_dict_wrapper
0
Ð0
Ñ1"
trackable_list_wrapper
.
Î	variables"
_generic_user_object
:  (2total
:  (2count
0
Ô0
Õ1"
trackable_list_wrapper
.
Ò	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
,:* 2Adam/conv2d/kernel/m
: 2Adam/conv2d/bias/m
.:,  2Adam/conv2d_1/kernel/m
 : 2Adam/conv2d_1/bias/m
.:, @2Adam/conv2d_2/kernel/m
 :@2Adam/conv2d_2/bias/m
.:,@@2Adam/conv2d_3/kernel/m
 :@2Adam/conv2d_3/bias/m
.:,@ 2Adam/conv2d_4/kernel/m
 : 2Adam/conv2d_4/bias/m
%:#
2Adam/dense/kernel/m
:2Adam/dense/bias/m
':%
2Adam/dense_1/kernel/m
 :2Adam/dense_1/bias/m
&:$	2Adam/dense_2/kernel/m
:2Adam/dense_2/bias/m
,:* 2Adam/conv2d/kernel/v
: 2Adam/conv2d/bias/v
.:,  2Adam/conv2d_1/kernel/v
 : 2Adam/conv2d_1/bias/v
.:, @2Adam/conv2d_2/kernel/v
 :@2Adam/conv2d_2/bias/v
.:,@@2Adam/conv2d_3/kernel/v
 :@2Adam/conv2d_3/bias/v
.:,@ 2Adam/conv2d_4/kernel/v
 : 2Adam/conv2d_4/bias/v
%:#
2Adam/dense/kernel/v
:2Adam/dense/bias/v
':%
2Adam/dense_1/kernel/v
 :2Adam/dense_1/bias/v
&:$	2Adam/dense_2/kernel/v
:2Adam/dense_2/bias/vø
!__inference__wrapped_model_142898ÒP Ý,-Þ/0ßFGàIJá`aâcdãopärså~æç£¤è¦§é½¾êÀÁë×ØìÚÛ?¢<
5¢2
0-
conv2d_inputÿÿÿÿÿÿÿÿÿææ
ª "=ª:
8
quant_dense_2'$
quant_dense_2ÿÿÿÿÿÿÿÿÿ<
__inference_loss_fn_0_147144Ý¢

¢ 
ª " <
__inference_loss_fn_1_147163ß¢

¢ 
ª " <
__inference_loss_fn_2_147182á¢

¢ 
ª " <
__inference_loss_fn_3_147191ã¢

¢ 
ª " <
__inference_loss_fn_4_147200å¢

¢ 
ª " <
__inference_loss_fn_5_147219ç¢

¢ 
ª " <
__inference_loss_fn_6_147228é¢

¢ 
ª " î
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_147173R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Æ
0__inference_max_pooling2d_1_layer_call_fn_147168R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿî
K__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_147210R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Æ
0__inference_max_pooling2d_2_layer_call_fn_147205R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿì
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_147154R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Ä
.__inference_max_pooling2d_layer_call_fn_147149R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÄ
J__inference_quant_conv2d_1_layer_call_and_return_conditional_losses_146278vßFGàIJ;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿKK 
p 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿKK 
 Ä
J__inference_quant_conv2d_1_layer_call_and_return_conditional_losses_146331vßFGàIJ;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿKK 
p
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿKK 
 
/__inference_quant_conv2d_1_layer_call_fn_146236ißFGàIJ;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿKK 
p 
ª " ÿÿÿÿÿÿÿÿÿKK 
/__inference_quant_conv2d_1_layer_call_fn_146253ißFGàIJ;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿKK 
p
ª " ÿÿÿÿÿÿÿÿÿKK Ä
J__inference_quant_conv2d_2_layer_call_and_return_conditional_losses_146414vá`aâcd;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ 
p 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@
 Ä
J__inference_quant_conv2d_2_layer_call_and_return_conditional_losses_146467vá`aâcd;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ 
p
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@
 
/__inference_quant_conv2d_2_layer_call_fn_146372iá`aâcd;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ 
p 
ª " ÿÿÿÿÿÿÿÿÿ@
/__inference_quant_conv2d_2_layer_call_fn_146389iá`aâcd;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ 
p
ª " ÿÿÿÿÿÿÿÿÿ@Ä
J__inference_quant_conv2d_3_layer_call_and_return_conditional_losses_146530vãopärs;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ@
p 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@
 Ä
J__inference_quant_conv2d_3_layer_call_and_return_conditional_losses_146583vãopärs;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ@
p
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@
 
/__inference_quant_conv2d_3_layer_call_fn_146488iãopärs;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ@
p 
ª " ÿÿÿÿÿÿÿÿÿ@
/__inference_quant_conv2d_3_layer_call_fn_146505iãopärs;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ@
p
ª " ÿÿÿÿÿÿÿÿÿ@Æ
J__inference_quant_conv2d_4_layer_call_and_return_conditional_losses_146646x
å~æ;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ@
p 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ 
 Æ
J__inference_quant_conv2d_4_layer_call_and_return_conditional_losses_146699x
å~æ;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ@
p
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ 
 
/__inference_quant_conv2d_4_layer_call_fn_146604k
å~æ;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ@
p 
ª " ÿÿÿÿÿÿÿÿÿ 
/__inference_quant_conv2d_4_layer_call_fn_146621k
å~æ;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ@
p
ª " ÿÿÿÿÿÿÿÿÿ Æ
H__inference_quant_conv2d_layer_call_and_return_conditional_losses_146142zÝ,-Þ/0=¢:
3¢0
*'
inputsÿÿÿÿÿÿÿÿÿææ
p 
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿââ 
 Æ
H__inference_quant_conv2d_layer_call_and_return_conditional_losses_146195zÝ,-Þ/0=¢:
3¢0
*'
inputsÿÿÿÿÿÿÿÿÿææ
p
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿââ 
 
-__inference_quant_conv2d_layer_call_fn_146100mÝ,-Þ/0=¢:
3¢0
*'
inputsÿÿÿÿÿÿÿÿÿææ
p 
ª ""ÿÿÿÿÿÿÿÿÿââ 
-__inference_quant_conv2d_layer_call_fn_146117mÝ,-Þ/0=¢:
3¢0
*'
inputsÿÿÿÿÿÿÿÿÿææ
p
ª ""ÿÿÿÿÿÿÿÿÿââ ¹
I__inference_quant_dense_1_layer_call_and_return_conditional_losses_146947lé½¾êÀÁ4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 ¹
I__inference_quant_dense_1_layer_call_and_return_conditional_losses_147000lé½¾êÀÁ4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 
.__inference_quant_dense_1_layer_call_fn_146905_é½¾êÀÁ4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "ÿÿÿÿÿÿÿÿÿ
.__inference_quant_dense_1_layer_call_fn_146922_é½¾êÀÁ4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p
ª "ÿÿÿÿÿÿÿÿÿ¸
I__inference_quant_dense_2_layer_call_and_return_conditional_losses_147086kë×ØìÚÛ4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ¸
I__inference_quant_dense_2_layer_call_and_return_conditional_losses_147135kë×ØìÚÛ4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
.__inference_quant_dense_2_layer_call_fn_147048^ë×ØìÚÛ4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "ÿÿÿÿÿÿÿÿÿ
.__inference_quant_dense_2_layer_call_fn_147065^ë×ØìÚÛ4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p
ª "ÿÿÿÿÿÿÿÿÿ·
G__inference_quant_dense_layer_call_and_return_conditional_losses_146804lç£¤è¦§4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 ·
G__inference_quant_dense_layer_call_and_return_conditional_losses_146857lç£¤è¦§4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 
,__inference_quant_dense_layer_call_fn_146762_ç£¤è¦§4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "ÿÿÿÿÿÿÿÿÿ
,__inference_quant_dense_layer_call_fn_146779_ç£¤è¦§4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p
ª "ÿÿÿÿÿÿÿÿÿ­
K__inference_quant_dropout_1_layer_call_and_return_conditional_losses_147019^4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 ­
K__inference_quant_dropout_1_layer_call_and_return_conditional_losses_147031^4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 
0__inference_quant_dropout_1_layer_call_fn_147009Q4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "ÿÿÿÿÿÿÿÿÿ
0__inference_quant_dropout_1_layer_call_fn_147014Q4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p
ª "ÿÿÿÿÿÿÿÿÿ«
I__inference_quant_dropout_layer_call_and_return_conditional_losses_146876^4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 «
I__inference_quant_dropout_layer_call_and_return_conditional_losses_146888^4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 
.__inference_quant_dropout_layer_call_fn_146866Q4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "ÿÿÿÿÿÿÿÿÿ
.__inference_quant_dropout_layer_call_fn_146871Q4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p
ª "ÿÿÿÿÿÿÿÿÿ²
I__inference_quant_flatten_layer_call_and_return_conditional_losses_146739e;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ 
p 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 ²
I__inference_quant_flatten_layer_call_and_return_conditional_losses_146745e;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ 
p
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 
.__inference_quant_flatten_layer_call_fn_146728X;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ 
p 
ª "ÿÿÿÿÿÿÿÿÿ
.__inference_quant_flatten_layer_call_fn_146733X;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ 
p
ª "ÿÿÿÿÿÿÿÿÿÁ
Q__inference_quant_max_pooling2d_1_layer_call_and_return_conditional_losses_146350l;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿKK 
p 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ 
 Á
Q__inference_quant_max_pooling2d_1_layer_call_and_return_conditional_losses_146355l;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿKK 
p
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ 
 
6__inference_quant_max_pooling2d_1_layer_call_fn_146340_;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿKK 
p 
ª " ÿÿÿÿÿÿÿÿÿ 
6__inference_quant_max_pooling2d_1_layer_call_fn_146345_;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿKK 
p
ª " ÿÿÿÿÿÿÿÿÿ Á
Q__inference_quant_max_pooling2d_2_layer_call_and_return_conditional_losses_146718l;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ 
p 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ 
 Á
Q__inference_quant_max_pooling2d_2_layer_call_and_return_conditional_losses_146723l;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ 
p
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ 
 
6__inference_quant_max_pooling2d_2_layer_call_fn_146708_;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ 
p 
ª " ÿÿÿÿÿÿÿÿÿ 
6__inference_quant_max_pooling2d_2_layer_call_fn_146713_;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ 
p
ª " ÿÿÿÿÿÿÿÿÿ Á
O__inference_quant_max_pooling2d_layer_call_and_return_conditional_losses_146214n=¢:
3¢0
*'
inputsÿÿÿÿÿÿÿÿÿââ 
p 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿKK 
 Á
O__inference_quant_max_pooling2d_layer_call_and_return_conditional_losses_146219n=¢:
3¢0
*'
inputsÿÿÿÿÿÿÿÿÿââ 
p
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿKK 
 
4__inference_quant_max_pooling2d_layer_call_fn_146204a=¢:
3¢0
*'
inputsÿÿÿÿÿÿÿÿÿââ 
p 
ª " ÿÿÿÿÿÿÿÿÿKK 
4__inference_quant_max_pooling2d_layer_call_fn_146209a=¢:
3¢0
*'
inputsÿÿÿÿÿÿÿÿÿââ 
p
ª " ÿÿÿÿÿÿÿÿÿKK Â
J__inference_quantize_layer_layer_call_and_return_conditional_losses_146062t =¢:
3¢0
*'
inputsÿÿÿÿÿÿÿÿÿææ
p 
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿææ
 Â
J__inference_quantize_layer_layer_call_and_return_conditional_losses_146083t =¢:
3¢0
*'
inputsÿÿÿÿÿÿÿÿÿææ
p
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿææ
 
/__inference_quantize_layer_layer_call_fn_146044g =¢:
3¢0
*'
inputsÿÿÿÿÿÿÿÿÿææ
p 
ª ""ÿÿÿÿÿÿÿÿÿææ
/__inference_quantize_layer_layer_call_fn_146053g =¢:
3¢0
*'
inputsÿÿÿÿÿÿÿÿÿææ
p
ª ""ÿÿÿÿÿÿÿÿÿææ
F__inference_sequential_layer_call_and_return_conditional_losses_144899ÂP Ý,-Þ/0ßFGàIJá`aâcdãopärså~æç£¤è¦§é½¾êÀÁë×ØìÚÛG¢D
=¢:
0-
conv2d_inputÿÿÿÿÿÿÿÿÿææ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
F__inference_sequential_layer_call_and_return_conditional_losses_145046ÂP Ý,-Þ/0ßFGàIJá`aâcdãopärså~æç£¤è¦§é½¾êÀÁë×ØìÚÛG¢D
=¢:
0-
conv2d_inputÿÿÿÿÿÿÿÿÿææ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
F__inference_sequential_layer_call_and_return_conditional_losses_145605¼P Ý,-Þ/0ßFGàIJá`aâcdãopärså~æç£¤è¦§é½¾êÀÁë×ØìÚÛA¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿææ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
F__inference_sequential_layer_call_and_return_conditional_losses_146035¼P Ý,-Þ/0ßFGàIJá`aâcdãopärså~æç£¤è¦§é½¾êÀÁë×ØìÚÛA¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿææ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 å
+__inference_sequential_layer_call_fn_143439µP Ý,-Þ/0ßFGàIJá`aâcdãopärså~æç£¤è¦§é½¾êÀÁë×ØìÚÛG¢D
=¢:
0-
conv2d_inputÿÿÿÿÿÿÿÿÿææ
p 

 
ª "ÿÿÿÿÿÿÿÿÿå
+__inference_sequential_layer_call_fn_144752µP Ý,-Þ/0ßFGàIJá`aâcdãopärså~æç£¤è¦§é½¾êÀÁë×ØìÚÛG¢D
=¢:
0-
conv2d_inputÿÿÿÿÿÿÿÿÿææ
p

 
ª "ÿÿÿÿÿÿÿÿÿß
+__inference_sequential_layer_call_fn_145320¯P Ý,-Þ/0ßFGàIJá`aâcdãopärså~æç£¤è¦§é½¾êÀÁë×ØìÚÛA¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿææ
p 

 
ª "ÿÿÿÿÿÿÿÿÿß
+__inference_sequential_layer_call_fn_145425¯P Ý,-Þ/0ßFGàIJá`aâcdãopärså~æç£¤è¦§é½¾êÀÁë×ØìÚÛA¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿææ
p

 
ª "ÿÿÿÿÿÿÿÿÿ
$__inference_signature_wrapper_145187âP Ý,-Þ/0ßFGàIJá`aâcdãopärså~æç£¤è¦§é½¾êÀÁë×ØìÚÛO¢L
¢ 
EªB
@
conv2d_input0-
conv2d_inputÿÿÿÿÿÿÿÿÿææ"=ª:
8
quant_dense_2'$
quant_dense_2ÿÿÿÿÿÿÿÿÿ