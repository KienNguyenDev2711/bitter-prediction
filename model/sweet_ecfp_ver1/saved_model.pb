��
��
D
AddV2
x"T
y"T
z"T"
Ttype:
2	��
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
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
�
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( �
?
Mul
x"T
y"T
z"T"
Ttype:
2	�
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
dtypetype�
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
.
Rsqrt
x"T
y"T"
Ttype:

2
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
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
0
Sigmoid
x"T
y"T"
Ttype:

2
�
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
executor_typestring ��
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
<
Sub
x"T
y"T
z"T"
Ttype:
2	
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.11.02v2.11.0-rc2-17-gd5b57ca93e58��	
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
b
count_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_2
[
count_2/Read/ReadVariableOpReadVariableOpcount_2*
_output_shapes
: *
dtype0
b
total_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_2
[
total_2/Read/ReadVariableOpReadVariableOptotal_2*
_output_shapes
: *
dtype0
�
Adam/v/dense_454/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/v/dense_454/bias
{
)Adam/v/dense_454/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_454/bias*
_output_shapes
:*
dtype0
�
Adam/m/dense_454/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/m/dense_454/bias
{
)Adam/m/dense_454/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_454/bias*
_output_shapes
:*
dtype0
�
Adam/v/dense_454/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *(
shared_nameAdam/v/dense_454/kernel
�
+Adam/v/dense_454/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_454/kernel*
_output_shapes

: *
dtype0
�
Adam/m/dense_454/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *(
shared_nameAdam/m/dense_454/kernel
�
+Adam/m/dense_454/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_454/kernel*
_output_shapes

: *
dtype0
�
"Adam/v/batch_normalization_90/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"Adam/v/batch_normalization_90/beta
�
6Adam/v/batch_normalization_90/beta/Read/ReadVariableOpReadVariableOp"Adam/v/batch_normalization_90/beta*
_output_shapes
: *
dtype0
�
"Adam/m/batch_normalization_90/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"Adam/m/batch_normalization_90/beta
�
6Adam/m/batch_normalization_90/beta/Read/ReadVariableOpReadVariableOp"Adam/m/batch_normalization_90/beta*
_output_shapes
: *
dtype0
�
#Adam/v/batch_normalization_90/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *4
shared_name%#Adam/v/batch_normalization_90/gamma
�
7Adam/v/batch_normalization_90/gamma/Read/ReadVariableOpReadVariableOp#Adam/v/batch_normalization_90/gamma*
_output_shapes
: *
dtype0
�
#Adam/m/batch_normalization_90/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *4
shared_name%#Adam/m/batch_normalization_90/gamma
�
7Adam/m/batch_normalization_90/gamma/Read/ReadVariableOpReadVariableOp#Adam/m/batch_normalization_90/gamma*
_output_shapes
: *
dtype0
�
Adam/v/dense_453/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/v/dense_453/bias
{
)Adam/v/dense_453/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_453/bias*
_output_shapes
: *
dtype0
�
Adam/m/dense_453/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/m/dense_453/bias
{
)Adam/m/dense_453/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_453/bias*
_output_shapes
: *
dtype0
�
Adam/v/dense_453/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@ *(
shared_nameAdam/v/dense_453/kernel
�
+Adam/v/dense_453/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_453/kernel*
_output_shapes

:@ *
dtype0
�
Adam/m/dense_453/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@ *(
shared_nameAdam/m/dense_453/kernel
�
+Adam/m/dense_453/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_453/kernel*
_output_shapes

:@ *
dtype0
�
Adam/v/dense_452/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/v/dense_452/bias
{
)Adam/v/dense_452/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_452/bias*
_output_shapes
:@*
dtype0
�
Adam/m/dense_452/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/m/dense_452/bias
{
)Adam/m/dense_452/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_452/bias*
_output_shapes
:@*
dtype0
�
Adam/v/dense_452/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�@*(
shared_nameAdam/v/dense_452/kernel
�
+Adam/v/dense_452/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_452/kernel*
_output_shapes
:	�@*
dtype0
�
Adam/m/dense_452/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�@*(
shared_nameAdam/m/dense_452/kernel
�
+Adam/m/dense_452/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_452/kernel*
_output_shapes
:	�@*
dtype0
�
Adam/v/dense_451/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*&
shared_nameAdam/v/dense_451/bias
|
)Adam/v/dense_451/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_451/bias*
_output_shapes	
:�*
dtype0
�
Adam/m/dense_451/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*&
shared_nameAdam/m/dense_451/bias
|
)Adam/m/dense_451/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_451/bias*
_output_shapes	
:�*
dtype0
�
Adam/v/dense_451/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*(
shared_nameAdam/v/dense_451/kernel
�
+Adam/v/dense_451/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_451/kernel* 
_output_shapes
:
��*
dtype0
�
Adam/m/dense_451/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*(
shared_nameAdam/m/dense_451/kernel
�
+Adam/m/dense_451/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_451/kernel* 
_output_shapes
:
��*
dtype0
�
Adam/v/dense_450/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*&
shared_nameAdam/v/dense_450/bias
|
)Adam/v/dense_450/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_450/bias*
_output_shapes	
:�*
dtype0
�
Adam/m/dense_450/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*&
shared_nameAdam/m/dense_450/bias
|
)Adam/m/dense_450/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_450/bias*
_output_shapes	
:�*
dtype0
�
Adam/v/dense_450/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*(
shared_nameAdam/v/dense_450/kernel
�
+Adam/v/dense_450/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_450/kernel* 
_output_shapes
:
��*
dtype0
�
Adam/m/dense_450/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*(
shared_nameAdam/m/dense_450/kernel
�
+Adam/m/dense_450/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_450/kernel* 
_output_shapes
:
��*
dtype0
n
learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namelearning_rate
g
!learning_rate/Read/ReadVariableOpReadVariableOplearning_rate*
_output_shapes
: *
dtype0
f
	iterationVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	iteration
_
iteration/Read/ReadVariableOpReadVariableOp	iteration*
_output_shapes
: *
dtype0	
t
dense_454/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_454/bias
m
"dense_454/bias/Read/ReadVariableOpReadVariableOpdense_454/bias*
_output_shapes
:*
dtype0
|
dense_454/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *!
shared_namedense_454/kernel
u
$dense_454/kernel/Read/ReadVariableOpReadVariableOpdense_454/kernel*
_output_shapes

: *
dtype0
�
&batch_normalization_90/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *7
shared_name(&batch_normalization_90/moving_variance
�
:batch_normalization_90/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_90/moving_variance*
_output_shapes
: *
dtype0
�
"batch_normalization_90/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"batch_normalization_90/moving_mean
�
6batch_normalization_90/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_90/moving_mean*
_output_shapes
: *
dtype0
�
batch_normalization_90/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namebatch_normalization_90/beta
�
/batch_normalization_90/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_90/beta*
_output_shapes
: *
dtype0
�
batch_normalization_90/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_namebatch_normalization_90/gamma
�
0batch_normalization_90/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_90/gamma*
_output_shapes
: *
dtype0
t
dense_453/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_453/bias
m
"dense_453/bias/Read/ReadVariableOpReadVariableOpdense_453/bias*
_output_shapes
: *
dtype0
|
dense_453/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@ *!
shared_namedense_453/kernel
u
$dense_453/kernel/Read/ReadVariableOpReadVariableOpdense_453/kernel*
_output_shapes

:@ *
dtype0
t
dense_452/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_452/bias
m
"dense_452/bias/Read/ReadVariableOpReadVariableOpdense_452/bias*
_output_shapes
:@*
dtype0
}
dense_452/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�@*!
shared_namedense_452/kernel
v
$dense_452/kernel/Read/ReadVariableOpReadVariableOpdense_452/kernel*
_output_shapes
:	�@*
dtype0
u
dense_451/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_namedense_451/bias
n
"dense_451/bias/Read/ReadVariableOpReadVariableOpdense_451/bias*
_output_shapes	
:�*
dtype0
~
dense_451/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*!
shared_namedense_451/kernel
w
$dense_451/kernel/Read/ReadVariableOpReadVariableOpdense_451/kernel* 
_output_shapes
:
��*
dtype0
u
dense_450/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_namedense_450/bias
n
"dense_450/bias/Read/ReadVariableOpReadVariableOpdense_450/bias*
_output_shapes	
:�*
dtype0
~
dense_450/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*!
shared_namedense_450/kernel
w
$dense_450/kernel/Read/ReadVariableOpReadVariableOpdense_450/kernel* 
_output_shapes
:
��*
dtype0
�
serving_default_dense_450_inputPlaceholder*(
_output_shapes
:����������*
dtype0*
shape:����������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_dense_450_inputdense_450/kerneldense_450/biasdense_451/kerneldense_451/biasdense_452/kerneldense_452/biasdense_453/kerneldense_453/bias&batch_normalization_90/moving_variancebatch_normalization_90/gamma"batch_normalization_90/moving_meanbatch_normalization_90/betadense_454/kerneldense_454/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*0
_read_only_resource_inputs
	
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *.
f)R'
%__inference_signature_wrapper_2139846

NoOpNoOp
�Q
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�Q
value�QB�Q B�Q
�
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
	variables
trainable_variables
	regularization_losses

	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures*
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias*
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias*
�
 	variables
!trainable_variables
"regularization_losses
#	keras_api
$__call__
*%&call_and_return_all_conditional_losses

&kernel
'bias*
�
(	variables
)trainable_variables
*regularization_losses
+	keras_api
,__call__
*-&call_and_return_all_conditional_losses

.kernel
/bias*
�
0	variables
1trainable_variables
2regularization_losses
3	keras_api
4__call__
*5&call_and_return_all_conditional_losses
6axis
	7gamma
8beta
9moving_mean
:moving_variance*
�
;	variables
<trainable_variables
=regularization_losses
>	keras_api
?__call__
*@&call_and_return_all_conditional_losses

Akernel
Bbias*
j
0
1
2
3
&4
'5
.6
/7
78
89
910
:11
A12
B13*
Z
0
1
2
3
&4
'5
.6
/7
78
89
A10
B11*
* 
�
Cnon_trainable_variables

Dlayers
Emetrics
Flayer_regularization_losses
Glayer_metrics
	variables
trainable_variables
	regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
Htrace_0
Itrace_1
Jtrace_2
Ktrace_3* 
6
Ltrace_0
Mtrace_1
Ntrace_2
Otrace_3* 
* 
�
P
_variables
Q_iterations
R_learning_rate
S_index_dict
T
_momentums
U_velocities
V_update_step_xla*

Wserving_default* 

0
1*

0
1*
* 
�
Xnon_trainable_variables

Ylayers
Zmetrics
[layer_regularization_losses
\layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

]trace_0* 

^trace_0* 
`Z
VARIABLE_VALUEdense_450/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEdense_450/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*
* 
�
_non_trainable_variables

`layers
ametrics
blayer_regularization_losses
clayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

dtrace_0* 

etrace_0* 
`Z
VARIABLE_VALUEdense_451/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEdense_451/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*

&0
'1*

&0
'1*
* 
�
fnon_trainable_variables

glayers
hmetrics
ilayer_regularization_losses
jlayer_metrics
 	variables
!trainable_variables
"regularization_losses
$__call__
*%&call_and_return_all_conditional_losses
&%"call_and_return_conditional_losses*

ktrace_0* 

ltrace_0* 
`Z
VARIABLE_VALUEdense_452/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEdense_452/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*

.0
/1*

.0
/1*
* 
�
mnon_trainable_variables

nlayers
ometrics
player_regularization_losses
qlayer_metrics
(	variables
)trainable_variables
*regularization_losses
,__call__
*-&call_and_return_all_conditional_losses
&-"call_and_return_conditional_losses*

rtrace_0* 

strace_0* 
`Z
VARIABLE_VALUEdense_453/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEdense_453/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*
 
70
81
92
:3*

70
81*
* 
�
tnon_trainable_variables

ulayers
vmetrics
wlayer_regularization_losses
xlayer_metrics
0	variables
1trainable_variables
2regularization_losses
4__call__
*5&call_and_return_all_conditional_losses
&5"call_and_return_conditional_losses*

ytrace_0
ztrace_1* 

{trace_0
|trace_1* 
* 
ke
VARIABLE_VALUEbatch_normalization_90/gamma5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEbatch_normalization_90/beta4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE"batch_normalization_90/moving_mean;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE&batch_normalization_90/moving_variance?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*

A0
B1*

A0
B1*
* 
�
}non_trainable_variables

~layers
metrics
 �layer_regularization_losses
�layer_metrics
;	variables
<trainable_variables
=regularization_losses
?__call__
*@&call_and_return_all_conditional_losses
&@"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
`Z
VARIABLE_VALUEdense_454/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEdense_454/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE*

90
:1*
.
0
1
2
3
4
5*

�0
�1
�2*
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
�
Q0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21
�22
�23
�24*
SM
VARIABLE_VALUE	iteration0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUElearning_rate3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
f
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11*
f
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11*
�
�trace_0
�trace_1
�trace_2
�trace_3
�trace_4
�trace_5
�trace_6
�trace_7
�trace_8
�trace_9
�trace_10
�trace_11* 
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

90
:1*
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
<
�	variables
�	keras_api

�total

�count*
M
�	variables
�	keras_api

�total

�count
�
_fn_kwargs*
M
�	variables
�	keras_api

�total

�count
�
_fn_kwargs*
b\
VARIABLE_VALUEAdam/m/dense_450/kernel1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/dense_450/kernel1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/dense_450/bias1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/dense_450/bias1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/dense_451/kernel1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/dense_451/kernel1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/dense_451/bias1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/dense_451/bias1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/dense_452/kernel1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEAdam/v/dense_452/kernel2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/m/dense_452/bias2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/v/dense_452/bias2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEAdam/m/dense_453/kernel2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEAdam/v/dense_453/kernel2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/m/dense_453/bias2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/v/dense_453/bias2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUE#Adam/m/batch_normalization_90/gamma2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUE#Adam/v/batch_normalization_90/gamma2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE"Adam/m/batch_normalization_90/beta2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE"Adam/v/batch_normalization_90/beta2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEAdam/m/dense_454/kernel2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEAdam/v/dense_454/kernel2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/m/dense_454/bias2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/v/dense_454/bias2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUE*
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

�0
�1*

�	variables*
UO
VARIABLE_VALUEtotal_24keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_24keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

�0
�1*

�	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 

�0
�1*

�	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$dense_450/kernel/Read/ReadVariableOp"dense_450/bias/Read/ReadVariableOp$dense_451/kernel/Read/ReadVariableOp"dense_451/bias/Read/ReadVariableOp$dense_452/kernel/Read/ReadVariableOp"dense_452/bias/Read/ReadVariableOp$dense_453/kernel/Read/ReadVariableOp"dense_453/bias/Read/ReadVariableOp0batch_normalization_90/gamma/Read/ReadVariableOp/batch_normalization_90/beta/Read/ReadVariableOp6batch_normalization_90/moving_mean/Read/ReadVariableOp:batch_normalization_90/moving_variance/Read/ReadVariableOp$dense_454/kernel/Read/ReadVariableOp"dense_454/bias/Read/ReadVariableOpiteration/Read/ReadVariableOp!learning_rate/Read/ReadVariableOp+Adam/m/dense_450/kernel/Read/ReadVariableOp+Adam/v/dense_450/kernel/Read/ReadVariableOp)Adam/m/dense_450/bias/Read/ReadVariableOp)Adam/v/dense_450/bias/Read/ReadVariableOp+Adam/m/dense_451/kernel/Read/ReadVariableOp+Adam/v/dense_451/kernel/Read/ReadVariableOp)Adam/m/dense_451/bias/Read/ReadVariableOp)Adam/v/dense_451/bias/Read/ReadVariableOp+Adam/m/dense_452/kernel/Read/ReadVariableOp+Adam/v/dense_452/kernel/Read/ReadVariableOp)Adam/m/dense_452/bias/Read/ReadVariableOp)Adam/v/dense_452/bias/Read/ReadVariableOp+Adam/m/dense_453/kernel/Read/ReadVariableOp+Adam/v/dense_453/kernel/Read/ReadVariableOp)Adam/m/dense_453/bias/Read/ReadVariableOp)Adam/v/dense_453/bias/Read/ReadVariableOp7Adam/m/batch_normalization_90/gamma/Read/ReadVariableOp7Adam/v/batch_normalization_90/gamma/Read/ReadVariableOp6Adam/m/batch_normalization_90/beta/Read/ReadVariableOp6Adam/v/batch_normalization_90/beta/Read/ReadVariableOp+Adam/m/dense_454/kernel/Read/ReadVariableOp+Adam/v/dense_454/kernel/Read/ReadVariableOp)Adam/m/dense_454/bias/Read/ReadVariableOp)Adam/v/dense_454/bias/Read/ReadVariableOptotal_2/Read/ReadVariableOpcount_2/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOpConst*;
Tin4
220	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *)
f$R"
 __inference__traced_save_2140437
�

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_450/kerneldense_450/biasdense_451/kerneldense_451/biasdense_452/kerneldense_452/biasdense_453/kerneldense_453/biasbatch_normalization_90/gammabatch_normalization_90/beta"batch_normalization_90/moving_mean&batch_normalization_90/moving_variancedense_454/kerneldense_454/bias	iterationlearning_rateAdam/m/dense_450/kernelAdam/v/dense_450/kernelAdam/m/dense_450/biasAdam/v/dense_450/biasAdam/m/dense_451/kernelAdam/v/dense_451/kernelAdam/m/dense_451/biasAdam/v/dense_451/biasAdam/m/dense_452/kernelAdam/v/dense_452/kernelAdam/m/dense_452/biasAdam/v/dense_452/biasAdam/m/dense_453/kernelAdam/v/dense_453/kernelAdam/m/dense_453/biasAdam/v/dense_453/bias#Adam/m/batch_normalization_90/gamma#Adam/v/batch_normalization_90/gamma"Adam/m/batch_normalization_90/beta"Adam/v/batch_normalization_90/betaAdam/m/dense_454/kernelAdam/v/dense_454/kernelAdam/m/dense_454/biasAdam/v/dense_454/biastotal_2count_2total_1count_1totalcount*:
Tin3
12/*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *,
f'R%
#__inference__traced_restore_2140585ݓ
�

�
F__inference_dense_451_layer_call_and_return_conditional_losses_2140136

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������W
SigmoidSigmoidBiasAdd:output:0*
T0*(
_output_shapes
:����������[
IdentityIdentitySigmoid:y:0^NoOp*
T0*(
_output_shapes
:����������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�%
�
S__inference_batch_normalization_90_layer_call_and_return_conditional_losses_2140256

inputs5
'assignmovingavg_readvariableop_resource: 7
)assignmovingavg_1_readvariableop_resource: 3
%batchnorm_mul_readvariableop_resource: /
!batchnorm_readvariableop_resource: 
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

: *
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

: �
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:��������� l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

: *
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
: *
dtype0�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
: x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
: �
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
: *
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
: ~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
: �
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
: P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: ~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:��������� h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
: v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:��������� b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:��������� �
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:��������� : : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
8__inference_batch_normalization_90_layer_call_fn_2140189

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *&
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *\
fWRU
S__inference_batch_normalization_90_layer_call_and_return_conditional_losses_2139355o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:��������� `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:��������� : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�

�
F__inference_dense_450_layer_call_and_return_conditional_losses_2140116

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:����������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
/__inference_sequential_90_layer_call_fn_2139733
dense_450_input
unknown:
��
	unknown_0:	�
	unknown_1:
��
	unknown_2:	�
	unknown_3:	�@
	unknown_4:@
	unknown_5:@ 
	unknown_6: 
	unknown_7: 
	unknown_8: 
	unknown_9: 

unknown_10: 

unknown_11: 

unknown_12:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCalldense_450_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*.
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *S
fNRL
J__inference_sequential_90_layer_call_and_return_conditional_losses_2139669o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:����������: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
(
_output_shapes
:����������
)
_user_specified_namedense_450_input
�
R
$__inference__update_step_xla_2140041
gradient
variable:
��*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*!
_input_shapes
:
��: *
	_noinline(:J F
 
_output_shapes
:
��
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
�

�
F__inference_dense_450_layer_call_and_return_conditional_losses_2139431

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:����������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�$
�
J__inference_sequential_90_layer_call_and_return_conditional_losses_2139515

inputs%
dense_450_2139432:
�� 
dense_450_2139434:	�%
dense_451_2139449:
�� 
dense_451_2139451:	�$
dense_452_2139466:	�@
dense_452_2139468:@#
dense_453_2139483:@ 
dense_453_2139485: ,
batch_normalization_90_2139488: ,
batch_normalization_90_2139490: ,
batch_normalization_90_2139492: ,
batch_normalization_90_2139494: #
dense_454_2139509: 
dense_454_2139511:
identity��.batch_normalization_90/StatefulPartitionedCall�!dense_450/StatefulPartitionedCall�!dense_451/StatefulPartitionedCall�!dense_452/StatefulPartitionedCall�!dense_453/StatefulPartitionedCall�!dense_454/StatefulPartitionedCall�
!dense_450/StatefulPartitionedCallStatefulPartitionedCallinputsdense_450_2139432dense_450_2139434*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *O
fJRH
F__inference_dense_450_layer_call_and_return_conditional_losses_2139431�
!dense_451/StatefulPartitionedCallStatefulPartitionedCall*dense_450/StatefulPartitionedCall:output:0dense_451_2139449dense_451_2139451*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *O
fJRH
F__inference_dense_451_layer_call_and_return_conditional_losses_2139448�
!dense_452/StatefulPartitionedCallStatefulPartitionedCall*dense_451/StatefulPartitionedCall:output:0dense_452_2139466dense_452_2139468*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *O
fJRH
F__inference_dense_452_layer_call_and_return_conditional_losses_2139465�
!dense_453/StatefulPartitionedCallStatefulPartitionedCall*dense_452/StatefulPartitionedCall:output:0dense_453_2139483dense_453_2139485*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *O
fJRH
F__inference_dense_453_layer_call_and_return_conditional_losses_2139482�
.batch_normalization_90/StatefulPartitionedCallStatefulPartitionedCall*dense_453/StatefulPartitionedCall:output:0batch_normalization_90_2139488batch_normalization_90_2139490batch_normalization_90_2139492batch_normalization_90_2139494*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *&
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *\
fWRU
S__inference_batch_normalization_90_layer_call_and_return_conditional_losses_2139355�
!dense_454/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_90/StatefulPartitionedCall:output:0dense_454_2139509dense_454_2139511*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *O
fJRH
F__inference_dense_454_layer_call_and_return_conditional_losses_2139508y
IdentityIdentity*dense_454/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp/^batch_normalization_90/StatefulPartitionedCall"^dense_450/StatefulPartitionedCall"^dense_451/StatefulPartitionedCall"^dense_452/StatefulPartitionedCall"^dense_453/StatefulPartitionedCall"^dense_454/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:����������: : : : : : : : : : : : : : 2`
.batch_normalization_90/StatefulPartitionedCall.batch_normalization_90/StatefulPartitionedCall2F
!dense_450/StatefulPartitionedCall!dense_450/StatefulPartitionedCall2F
!dense_451/StatefulPartitionedCall!dense_451/StatefulPartitionedCall2F
!dense_452/StatefulPartitionedCall!dense_452/StatefulPartitionedCall2F
!dense_453/StatefulPartitionedCall!dense_453/StatefulPartitionedCall2F
!dense_454/StatefulPartitionedCall!dense_454/StatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
L
$__inference__update_step_xla_2140066
gradient
variable:@*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:@: *
	_noinline(:D @

_output_shapes
:@
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
�
L
$__inference__update_step_xla_2140096
gradient
variable:*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:: *
	_noinline(:D @

_output_shapes
:
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
�
L
$__inference__update_step_xla_2140081
gradient
variable: *
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

: : *
	_noinline(:D @

_output_shapes
: 
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
�

�
F__inference_dense_451_layer_call_and_return_conditional_losses_2139448

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������W
SigmoidSigmoidBiasAdd:output:0*
T0*(
_output_shapes
:����������[
IdentityIdentitySigmoid:y:0^NoOp*
T0*(
_output_shapes
:����������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
F__inference_dense_453_layer_call_and_return_conditional_losses_2139482

inputs0
matmul_readvariableop_resource:@ -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@ *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:��������� Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:��������� w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
+__inference_dense_451_layer_call_fn_2140125

inputs
unknown:
��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *O
fJRH
F__inference_dense_451_layer_call_and_return_conditional_losses_2139448p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
S__inference_batch_normalization_90_layer_call_and_return_conditional_losses_2139355

inputs/
!batchnorm_readvariableop_resource: 3
%batchnorm_mul_readvariableop_resource: 1
#batchnorm_readvariableop_1_resource: 1
#batchnorm_readvariableop_2_resource: 
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
: P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: ~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:��������� z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
: z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:��������� b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:��������� �
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:��������� : : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�%
�
S__inference_batch_normalization_90_layer_call_and_return_conditional_losses_2139402

inputs5
'assignmovingavg_readvariableop_resource: 7
)assignmovingavg_1_readvariableop_resource: 3
%batchnorm_mul_readvariableop_resource: /
!batchnorm_readvariableop_resource: 
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

: *
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

: �
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:��������� l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

: *
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
: *
dtype0�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
: x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
: �
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
: *
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
: ~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
: �
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
: P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: ~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:��������� h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
: v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:��������� b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:��������� �
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:��������� : : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
/__inference_sequential_90_layer_call_fn_2139546
dense_450_input
unknown:
��
	unknown_0:	�
	unknown_1:
��
	unknown_2:	�
	unknown_3:	�@
	unknown_4:@
	unknown_5:@ 
	unknown_6: 
	unknown_7: 
	unknown_8: 
	unknown_9: 

unknown_10: 

unknown_11: 

unknown_12:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCalldense_450_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*0
_read_only_resource_inputs
	
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *S
fNRL
J__inference_sequential_90_layer_call_and_return_conditional_losses_2139515o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:����������: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
(
_output_shapes
:����������
)
_user_specified_namedense_450_input
�W
�
"__inference__wrapped_model_2139331
dense_450_inputJ
6sequential_90_dense_450_matmul_readvariableop_resource:
��F
7sequential_90_dense_450_biasadd_readvariableop_resource:	�J
6sequential_90_dense_451_matmul_readvariableop_resource:
��F
7sequential_90_dense_451_biasadd_readvariableop_resource:	�I
6sequential_90_dense_452_matmul_readvariableop_resource:	�@E
7sequential_90_dense_452_biasadd_readvariableop_resource:@H
6sequential_90_dense_453_matmul_readvariableop_resource:@ E
7sequential_90_dense_453_biasadd_readvariableop_resource: T
Fsequential_90_batch_normalization_90_batchnorm_readvariableop_resource: X
Jsequential_90_batch_normalization_90_batchnorm_mul_readvariableop_resource: V
Hsequential_90_batch_normalization_90_batchnorm_readvariableop_1_resource: V
Hsequential_90_batch_normalization_90_batchnorm_readvariableop_2_resource: H
6sequential_90_dense_454_matmul_readvariableop_resource: E
7sequential_90_dense_454_biasadd_readvariableop_resource:
identity��=sequential_90/batch_normalization_90/batchnorm/ReadVariableOp�?sequential_90/batch_normalization_90/batchnorm/ReadVariableOp_1�?sequential_90/batch_normalization_90/batchnorm/ReadVariableOp_2�Asequential_90/batch_normalization_90/batchnorm/mul/ReadVariableOp�.sequential_90/dense_450/BiasAdd/ReadVariableOp�-sequential_90/dense_450/MatMul/ReadVariableOp�.sequential_90/dense_451/BiasAdd/ReadVariableOp�-sequential_90/dense_451/MatMul/ReadVariableOp�.sequential_90/dense_452/BiasAdd/ReadVariableOp�-sequential_90/dense_452/MatMul/ReadVariableOp�.sequential_90/dense_453/BiasAdd/ReadVariableOp�-sequential_90/dense_453/MatMul/ReadVariableOp�.sequential_90/dense_454/BiasAdd/ReadVariableOp�-sequential_90/dense_454/MatMul/ReadVariableOp�
-sequential_90/dense_450/MatMul/ReadVariableOpReadVariableOp6sequential_90_dense_450_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
sequential_90/dense_450/MatMulMatMuldense_450_input5sequential_90/dense_450/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
.sequential_90/dense_450/BiasAdd/ReadVariableOpReadVariableOp7sequential_90_dense_450_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
sequential_90/dense_450/BiasAddBiasAdd(sequential_90/dense_450/MatMul:product:06sequential_90/dense_450/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
sequential_90/dense_450/ReluRelu(sequential_90/dense_450/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
-sequential_90/dense_451/MatMul/ReadVariableOpReadVariableOp6sequential_90_dense_451_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
sequential_90/dense_451/MatMulMatMul*sequential_90/dense_450/Relu:activations:05sequential_90/dense_451/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
.sequential_90/dense_451/BiasAdd/ReadVariableOpReadVariableOp7sequential_90_dense_451_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
sequential_90/dense_451/BiasAddBiasAdd(sequential_90/dense_451/MatMul:product:06sequential_90/dense_451/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
sequential_90/dense_451/SigmoidSigmoid(sequential_90/dense_451/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
-sequential_90/dense_452/MatMul/ReadVariableOpReadVariableOp6sequential_90_dense_452_matmul_readvariableop_resource*
_output_shapes
:	�@*
dtype0�
sequential_90/dense_452/MatMulMatMul#sequential_90/dense_451/Sigmoid:y:05sequential_90/dense_452/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
.sequential_90/dense_452/BiasAdd/ReadVariableOpReadVariableOp7sequential_90_dense_452_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
sequential_90/dense_452/BiasAddBiasAdd(sequential_90/dense_452/MatMul:product:06sequential_90/dense_452/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
sequential_90/dense_452/SigmoidSigmoid(sequential_90/dense_452/BiasAdd:output:0*
T0*'
_output_shapes
:���������@�
-sequential_90/dense_453/MatMul/ReadVariableOpReadVariableOp6sequential_90_dense_453_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype0�
sequential_90/dense_453/MatMulMatMul#sequential_90/dense_452/Sigmoid:y:05sequential_90/dense_453/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
.sequential_90/dense_453/BiasAdd/ReadVariableOpReadVariableOp7sequential_90_dense_453_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
sequential_90/dense_453/BiasAddBiasAdd(sequential_90/dense_453/MatMul:product:06sequential_90/dense_453/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
sequential_90/dense_453/SigmoidSigmoid(sequential_90/dense_453/BiasAdd:output:0*
T0*'
_output_shapes
:��������� �
=sequential_90/batch_normalization_90/batchnorm/ReadVariableOpReadVariableOpFsequential_90_batch_normalization_90_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype0y
4sequential_90/batch_normalization_90/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
2sequential_90/batch_normalization_90/batchnorm/addAddV2Esequential_90/batch_normalization_90/batchnorm/ReadVariableOp:value:0=sequential_90/batch_normalization_90/batchnorm/add/y:output:0*
T0*
_output_shapes
: �
4sequential_90/batch_normalization_90/batchnorm/RsqrtRsqrt6sequential_90/batch_normalization_90/batchnorm/add:z:0*
T0*
_output_shapes
: �
Asequential_90/batch_normalization_90/batchnorm/mul/ReadVariableOpReadVariableOpJsequential_90_batch_normalization_90_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype0�
2sequential_90/batch_normalization_90/batchnorm/mulMul8sequential_90/batch_normalization_90/batchnorm/Rsqrt:y:0Isequential_90/batch_normalization_90/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: �
4sequential_90/batch_normalization_90/batchnorm/mul_1Mul#sequential_90/dense_453/Sigmoid:y:06sequential_90/batch_normalization_90/batchnorm/mul:z:0*
T0*'
_output_shapes
:��������� �
?sequential_90/batch_normalization_90/batchnorm/ReadVariableOp_1ReadVariableOpHsequential_90_batch_normalization_90_batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype0�
4sequential_90/batch_normalization_90/batchnorm/mul_2MulGsequential_90/batch_normalization_90/batchnorm/ReadVariableOp_1:value:06sequential_90/batch_normalization_90/batchnorm/mul:z:0*
T0*
_output_shapes
: �
?sequential_90/batch_normalization_90/batchnorm/ReadVariableOp_2ReadVariableOpHsequential_90_batch_normalization_90_batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype0�
2sequential_90/batch_normalization_90/batchnorm/subSubGsequential_90/batch_normalization_90/batchnorm/ReadVariableOp_2:value:08sequential_90/batch_normalization_90/batchnorm/mul_2:z:0*
T0*
_output_shapes
: �
4sequential_90/batch_normalization_90/batchnorm/add_1AddV28sequential_90/batch_normalization_90/batchnorm/mul_1:z:06sequential_90/batch_normalization_90/batchnorm/sub:z:0*
T0*'
_output_shapes
:��������� �
-sequential_90/dense_454/MatMul/ReadVariableOpReadVariableOp6sequential_90_dense_454_matmul_readvariableop_resource*
_output_shapes

: *
dtype0�
sequential_90/dense_454/MatMulMatMul8sequential_90/batch_normalization_90/batchnorm/add_1:z:05sequential_90/dense_454/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
.sequential_90/dense_454/BiasAdd/ReadVariableOpReadVariableOp7sequential_90_dense_454_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
sequential_90/dense_454/BiasAddBiasAdd(sequential_90/dense_454/MatMul:product:06sequential_90/dense_454/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
sequential_90/dense_454/SigmoidSigmoid(sequential_90/dense_454/BiasAdd:output:0*
T0*'
_output_shapes
:���������r
IdentityIdentity#sequential_90/dense_454/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp>^sequential_90/batch_normalization_90/batchnorm/ReadVariableOp@^sequential_90/batch_normalization_90/batchnorm/ReadVariableOp_1@^sequential_90/batch_normalization_90/batchnorm/ReadVariableOp_2B^sequential_90/batch_normalization_90/batchnorm/mul/ReadVariableOp/^sequential_90/dense_450/BiasAdd/ReadVariableOp.^sequential_90/dense_450/MatMul/ReadVariableOp/^sequential_90/dense_451/BiasAdd/ReadVariableOp.^sequential_90/dense_451/MatMul/ReadVariableOp/^sequential_90/dense_452/BiasAdd/ReadVariableOp.^sequential_90/dense_452/MatMul/ReadVariableOp/^sequential_90/dense_453/BiasAdd/ReadVariableOp.^sequential_90/dense_453/MatMul/ReadVariableOp/^sequential_90/dense_454/BiasAdd/ReadVariableOp.^sequential_90/dense_454/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:����������: : : : : : : : : : : : : : 2~
=sequential_90/batch_normalization_90/batchnorm/ReadVariableOp=sequential_90/batch_normalization_90/batchnorm/ReadVariableOp2�
?sequential_90/batch_normalization_90/batchnorm/ReadVariableOp_1?sequential_90/batch_normalization_90/batchnorm/ReadVariableOp_12�
?sequential_90/batch_normalization_90/batchnorm/ReadVariableOp_2?sequential_90/batch_normalization_90/batchnorm/ReadVariableOp_22�
Asequential_90/batch_normalization_90/batchnorm/mul/ReadVariableOpAsequential_90/batch_normalization_90/batchnorm/mul/ReadVariableOp2`
.sequential_90/dense_450/BiasAdd/ReadVariableOp.sequential_90/dense_450/BiasAdd/ReadVariableOp2^
-sequential_90/dense_450/MatMul/ReadVariableOp-sequential_90/dense_450/MatMul/ReadVariableOp2`
.sequential_90/dense_451/BiasAdd/ReadVariableOp.sequential_90/dense_451/BiasAdd/ReadVariableOp2^
-sequential_90/dense_451/MatMul/ReadVariableOp-sequential_90/dense_451/MatMul/ReadVariableOp2`
.sequential_90/dense_452/BiasAdd/ReadVariableOp.sequential_90/dense_452/BiasAdd/ReadVariableOp2^
-sequential_90/dense_452/MatMul/ReadVariableOp-sequential_90/dense_452/MatMul/ReadVariableOp2`
.sequential_90/dense_453/BiasAdd/ReadVariableOp.sequential_90/dense_453/BiasAdd/ReadVariableOp2^
-sequential_90/dense_453/MatMul/ReadVariableOp-sequential_90/dense_453/MatMul/ReadVariableOp2`
.sequential_90/dense_454/BiasAdd/ReadVariableOp.sequential_90/dense_454/BiasAdd/ReadVariableOp2^
-sequential_90/dense_454/MatMul/ReadVariableOp-sequential_90/dense_454/MatMul/ReadVariableOp:Y U
(
_output_shapes
:����������
)
_user_specified_namedense_450_input
�$
�
J__inference_sequential_90_layer_call_and_return_conditional_losses_2139771
dense_450_input%
dense_450_2139736:
�� 
dense_450_2139738:	�%
dense_451_2139741:
�� 
dense_451_2139743:	�$
dense_452_2139746:	�@
dense_452_2139748:@#
dense_453_2139751:@ 
dense_453_2139753: ,
batch_normalization_90_2139756: ,
batch_normalization_90_2139758: ,
batch_normalization_90_2139760: ,
batch_normalization_90_2139762: #
dense_454_2139765: 
dense_454_2139767:
identity��.batch_normalization_90/StatefulPartitionedCall�!dense_450/StatefulPartitionedCall�!dense_451/StatefulPartitionedCall�!dense_452/StatefulPartitionedCall�!dense_453/StatefulPartitionedCall�!dense_454/StatefulPartitionedCall�
!dense_450/StatefulPartitionedCallStatefulPartitionedCalldense_450_inputdense_450_2139736dense_450_2139738*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *O
fJRH
F__inference_dense_450_layer_call_and_return_conditional_losses_2139431�
!dense_451/StatefulPartitionedCallStatefulPartitionedCall*dense_450/StatefulPartitionedCall:output:0dense_451_2139741dense_451_2139743*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *O
fJRH
F__inference_dense_451_layer_call_and_return_conditional_losses_2139448�
!dense_452/StatefulPartitionedCallStatefulPartitionedCall*dense_451/StatefulPartitionedCall:output:0dense_452_2139746dense_452_2139748*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *O
fJRH
F__inference_dense_452_layer_call_and_return_conditional_losses_2139465�
!dense_453/StatefulPartitionedCallStatefulPartitionedCall*dense_452/StatefulPartitionedCall:output:0dense_453_2139751dense_453_2139753*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *O
fJRH
F__inference_dense_453_layer_call_and_return_conditional_losses_2139482�
.batch_normalization_90/StatefulPartitionedCallStatefulPartitionedCall*dense_453/StatefulPartitionedCall:output:0batch_normalization_90_2139756batch_normalization_90_2139758batch_normalization_90_2139760batch_normalization_90_2139762*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *&
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *\
fWRU
S__inference_batch_normalization_90_layer_call_and_return_conditional_losses_2139355�
!dense_454/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_90/StatefulPartitionedCall:output:0dense_454_2139765dense_454_2139767*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *O
fJRH
F__inference_dense_454_layer_call_and_return_conditional_losses_2139508y
IdentityIdentity*dense_454/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp/^batch_normalization_90/StatefulPartitionedCall"^dense_450/StatefulPartitionedCall"^dense_451/StatefulPartitionedCall"^dense_452/StatefulPartitionedCall"^dense_453/StatefulPartitionedCall"^dense_454/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:����������: : : : : : : : : : : : : : 2`
.batch_normalization_90/StatefulPartitionedCall.batch_normalization_90/StatefulPartitionedCall2F
!dense_450/StatefulPartitionedCall!dense_450/StatefulPartitionedCall2F
!dense_451/StatefulPartitionedCall!dense_451/StatefulPartitionedCall2F
!dense_452/StatefulPartitionedCall!dense_452/StatefulPartitionedCall2F
!dense_453/StatefulPartitionedCall!dense_453/StatefulPartitionedCall2F
!dense_454/StatefulPartitionedCall!dense_454/StatefulPartitionedCall:Y U
(
_output_shapes
:����������
)
_user_specified_namedense_450_input
�

�
F__inference_dense_452_layer_call_and_return_conditional_losses_2139465

inputs1
matmul_readvariableop_resource:	�@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:���������@Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:���������@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
F__inference_dense_454_layer_call_and_return_conditional_losses_2140276

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:���������Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:��������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�$
�
J__inference_sequential_90_layer_call_and_return_conditional_losses_2139669

inputs%
dense_450_2139634:
�� 
dense_450_2139636:	�%
dense_451_2139639:
�� 
dense_451_2139641:	�$
dense_452_2139644:	�@
dense_452_2139646:@#
dense_453_2139649:@ 
dense_453_2139651: ,
batch_normalization_90_2139654: ,
batch_normalization_90_2139656: ,
batch_normalization_90_2139658: ,
batch_normalization_90_2139660: #
dense_454_2139663: 
dense_454_2139665:
identity��.batch_normalization_90/StatefulPartitionedCall�!dense_450/StatefulPartitionedCall�!dense_451/StatefulPartitionedCall�!dense_452/StatefulPartitionedCall�!dense_453/StatefulPartitionedCall�!dense_454/StatefulPartitionedCall�
!dense_450/StatefulPartitionedCallStatefulPartitionedCallinputsdense_450_2139634dense_450_2139636*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *O
fJRH
F__inference_dense_450_layer_call_and_return_conditional_losses_2139431�
!dense_451/StatefulPartitionedCallStatefulPartitionedCall*dense_450/StatefulPartitionedCall:output:0dense_451_2139639dense_451_2139641*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *O
fJRH
F__inference_dense_451_layer_call_and_return_conditional_losses_2139448�
!dense_452/StatefulPartitionedCallStatefulPartitionedCall*dense_451/StatefulPartitionedCall:output:0dense_452_2139644dense_452_2139646*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *O
fJRH
F__inference_dense_452_layer_call_and_return_conditional_losses_2139465�
!dense_453/StatefulPartitionedCallStatefulPartitionedCall*dense_452/StatefulPartitionedCall:output:0dense_453_2139649dense_453_2139651*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *O
fJRH
F__inference_dense_453_layer_call_and_return_conditional_losses_2139482�
.batch_normalization_90/StatefulPartitionedCallStatefulPartitionedCall*dense_453/StatefulPartitionedCall:output:0batch_normalization_90_2139654batch_normalization_90_2139656batch_normalization_90_2139658batch_normalization_90_2139660*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *\
fWRU
S__inference_batch_normalization_90_layer_call_and_return_conditional_losses_2139402�
!dense_454/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_90/StatefulPartitionedCall:output:0dense_454_2139663dense_454_2139665*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *O
fJRH
F__inference_dense_454_layer_call_and_return_conditional_losses_2139508y
IdentityIdentity*dense_454/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp/^batch_normalization_90/StatefulPartitionedCall"^dense_450/StatefulPartitionedCall"^dense_451/StatefulPartitionedCall"^dense_452/StatefulPartitionedCall"^dense_453/StatefulPartitionedCall"^dense_454/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:����������: : : : : : : : : : : : : : 2`
.batch_normalization_90/StatefulPartitionedCall.batch_normalization_90/StatefulPartitionedCall2F
!dense_450/StatefulPartitionedCall!dense_450/StatefulPartitionedCall2F
!dense_451/StatefulPartitionedCall!dense_451/StatefulPartitionedCall2F
!dense_452/StatefulPartitionedCall!dense_452/StatefulPartitionedCall2F
!dense_453/StatefulPartitionedCall!dense_453/StatefulPartitionedCall2F
!dense_454/StatefulPartitionedCall!dense_454/StatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
+__inference_dense_450_layer_call_fn_2140105

inputs
unknown:
��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *O
fJRH
F__inference_dense_450_layer_call_and_return_conditional_losses_2139431p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�`
�
J__inference_sequential_90_layer_call_and_return_conditional_losses_2140036

inputs<
(dense_450_matmul_readvariableop_resource:
��8
)dense_450_biasadd_readvariableop_resource:	�<
(dense_451_matmul_readvariableop_resource:
��8
)dense_451_biasadd_readvariableop_resource:	�;
(dense_452_matmul_readvariableop_resource:	�@7
)dense_452_biasadd_readvariableop_resource:@:
(dense_453_matmul_readvariableop_resource:@ 7
)dense_453_biasadd_readvariableop_resource: L
>batch_normalization_90_assignmovingavg_readvariableop_resource: N
@batch_normalization_90_assignmovingavg_1_readvariableop_resource: J
<batch_normalization_90_batchnorm_mul_readvariableop_resource: F
8batch_normalization_90_batchnorm_readvariableop_resource: :
(dense_454_matmul_readvariableop_resource: 7
)dense_454_biasadd_readvariableop_resource:
identity��&batch_normalization_90/AssignMovingAvg�5batch_normalization_90/AssignMovingAvg/ReadVariableOp�(batch_normalization_90/AssignMovingAvg_1�7batch_normalization_90/AssignMovingAvg_1/ReadVariableOp�/batch_normalization_90/batchnorm/ReadVariableOp�3batch_normalization_90/batchnorm/mul/ReadVariableOp� dense_450/BiasAdd/ReadVariableOp�dense_450/MatMul/ReadVariableOp� dense_451/BiasAdd/ReadVariableOp�dense_451/MatMul/ReadVariableOp� dense_452/BiasAdd/ReadVariableOp�dense_452/MatMul/ReadVariableOp� dense_453/BiasAdd/ReadVariableOp�dense_453/MatMul/ReadVariableOp� dense_454/BiasAdd/ReadVariableOp�dense_454/MatMul/ReadVariableOp�
dense_450/MatMul/ReadVariableOpReadVariableOp(dense_450_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0~
dense_450/MatMulMatMulinputs'dense_450/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
 dense_450/BiasAdd/ReadVariableOpReadVariableOp)dense_450_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_450/BiasAddBiasAdddense_450/MatMul:product:0(dense_450/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������e
dense_450/ReluReludense_450/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
dense_451/MatMul/ReadVariableOpReadVariableOp(dense_451_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
dense_451/MatMulMatMuldense_450/Relu:activations:0'dense_451/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
 dense_451/BiasAdd/ReadVariableOpReadVariableOp)dense_451_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_451/BiasAddBiasAdddense_451/MatMul:product:0(dense_451/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������k
dense_451/SigmoidSigmoiddense_451/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
dense_452/MatMul/ReadVariableOpReadVariableOp(dense_452_matmul_readvariableop_resource*
_output_shapes
:	�@*
dtype0�
dense_452/MatMulMatMuldense_451/Sigmoid:y:0'dense_452/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
 dense_452/BiasAdd/ReadVariableOpReadVariableOp)dense_452_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
dense_452/BiasAddBiasAdddense_452/MatMul:product:0(dense_452/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@j
dense_452/SigmoidSigmoiddense_452/BiasAdd:output:0*
T0*'
_output_shapes
:���������@�
dense_453/MatMul/ReadVariableOpReadVariableOp(dense_453_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype0�
dense_453/MatMulMatMuldense_452/Sigmoid:y:0'dense_453/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
 dense_453/BiasAdd/ReadVariableOpReadVariableOp)dense_453_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
dense_453/BiasAddBiasAdddense_453/MatMul:product:0(dense_453/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� j
dense_453/SigmoidSigmoiddense_453/BiasAdd:output:0*
T0*'
_output_shapes
:��������� 
5batch_normalization_90/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
#batch_normalization_90/moments/meanMeandense_453/Sigmoid:y:0>batch_normalization_90/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

: *
	keep_dims(�
+batch_normalization_90/moments/StopGradientStopGradient,batch_normalization_90/moments/mean:output:0*
T0*
_output_shapes

: �
0batch_normalization_90/moments/SquaredDifferenceSquaredDifferencedense_453/Sigmoid:y:04batch_normalization_90/moments/StopGradient:output:0*
T0*'
_output_shapes
:��������� �
9batch_normalization_90/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
'batch_normalization_90/moments/varianceMean4batch_normalization_90/moments/SquaredDifference:z:0Bbatch_normalization_90/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

: *
	keep_dims(�
&batch_normalization_90/moments/SqueezeSqueeze,batch_normalization_90/moments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 �
(batch_normalization_90/moments/Squeeze_1Squeeze0batch_normalization_90/moments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 q
,batch_normalization_90/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
5batch_normalization_90/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_90_assignmovingavg_readvariableop_resource*
_output_shapes
: *
dtype0�
*batch_normalization_90/AssignMovingAvg/subSub=batch_normalization_90/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_90/moments/Squeeze:output:0*
T0*
_output_shapes
: �
*batch_normalization_90/AssignMovingAvg/mulMul.batch_normalization_90/AssignMovingAvg/sub:z:05batch_normalization_90/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
: �
&batch_normalization_90/AssignMovingAvgAssignSubVariableOp>batch_normalization_90_assignmovingavg_readvariableop_resource.batch_normalization_90/AssignMovingAvg/mul:z:06^batch_normalization_90/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0s
.batch_normalization_90/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
7batch_normalization_90/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_90_assignmovingavg_1_readvariableop_resource*
_output_shapes
: *
dtype0�
,batch_normalization_90/AssignMovingAvg_1/subSub?batch_normalization_90/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_90/moments/Squeeze_1:output:0*
T0*
_output_shapes
: �
,batch_normalization_90/AssignMovingAvg_1/mulMul0batch_normalization_90/AssignMovingAvg_1/sub:z:07batch_normalization_90/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
: �
(batch_normalization_90/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_90_assignmovingavg_1_readvariableop_resource0batch_normalization_90/AssignMovingAvg_1/mul:z:08^batch_normalization_90/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0k
&batch_normalization_90/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
$batch_normalization_90/batchnorm/addAddV21batch_normalization_90/moments/Squeeze_1:output:0/batch_normalization_90/batchnorm/add/y:output:0*
T0*
_output_shapes
: ~
&batch_normalization_90/batchnorm/RsqrtRsqrt(batch_normalization_90/batchnorm/add:z:0*
T0*
_output_shapes
: �
3batch_normalization_90/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_90_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype0�
$batch_normalization_90/batchnorm/mulMul*batch_normalization_90/batchnorm/Rsqrt:y:0;batch_normalization_90/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: �
&batch_normalization_90/batchnorm/mul_1Muldense_453/Sigmoid:y:0(batch_normalization_90/batchnorm/mul:z:0*
T0*'
_output_shapes
:��������� �
&batch_normalization_90/batchnorm/mul_2Mul/batch_normalization_90/moments/Squeeze:output:0(batch_normalization_90/batchnorm/mul:z:0*
T0*
_output_shapes
: �
/batch_normalization_90/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_90_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype0�
$batch_normalization_90/batchnorm/subSub7batch_normalization_90/batchnorm/ReadVariableOp:value:0*batch_normalization_90/batchnorm/mul_2:z:0*
T0*
_output_shapes
: �
&batch_normalization_90/batchnorm/add_1AddV2*batch_normalization_90/batchnorm/mul_1:z:0(batch_normalization_90/batchnorm/sub:z:0*
T0*'
_output_shapes
:��������� �
dense_454/MatMul/ReadVariableOpReadVariableOp(dense_454_matmul_readvariableop_resource*
_output_shapes

: *
dtype0�
dense_454/MatMulMatMul*batch_normalization_90/batchnorm/add_1:z:0'dense_454/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
 dense_454/BiasAdd/ReadVariableOpReadVariableOp)dense_454_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_454/BiasAddBiasAdddense_454/MatMul:product:0(dense_454/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������j
dense_454/SigmoidSigmoiddense_454/BiasAdd:output:0*
T0*'
_output_shapes
:���������d
IdentityIdentitydense_454/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp'^batch_normalization_90/AssignMovingAvg6^batch_normalization_90/AssignMovingAvg/ReadVariableOp)^batch_normalization_90/AssignMovingAvg_18^batch_normalization_90/AssignMovingAvg_1/ReadVariableOp0^batch_normalization_90/batchnorm/ReadVariableOp4^batch_normalization_90/batchnorm/mul/ReadVariableOp!^dense_450/BiasAdd/ReadVariableOp ^dense_450/MatMul/ReadVariableOp!^dense_451/BiasAdd/ReadVariableOp ^dense_451/MatMul/ReadVariableOp!^dense_452/BiasAdd/ReadVariableOp ^dense_452/MatMul/ReadVariableOp!^dense_453/BiasAdd/ReadVariableOp ^dense_453/MatMul/ReadVariableOp!^dense_454/BiasAdd/ReadVariableOp ^dense_454/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:����������: : : : : : : : : : : : : : 2P
&batch_normalization_90/AssignMovingAvg&batch_normalization_90/AssignMovingAvg2n
5batch_normalization_90/AssignMovingAvg/ReadVariableOp5batch_normalization_90/AssignMovingAvg/ReadVariableOp2T
(batch_normalization_90/AssignMovingAvg_1(batch_normalization_90/AssignMovingAvg_12r
7batch_normalization_90/AssignMovingAvg_1/ReadVariableOp7batch_normalization_90/AssignMovingAvg_1/ReadVariableOp2b
/batch_normalization_90/batchnorm/ReadVariableOp/batch_normalization_90/batchnorm/ReadVariableOp2j
3batch_normalization_90/batchnorm/mul/ReadVariableOp3batch_normalization_90/batchnorm/mul/ReadVariableOp2D
 dense_450/BiasAdd/ReadVariableOp dense_450/BiasAdd/ReadVariableOp2B
dense_450/MatMul/ReadVariableOpdense_450/MatMul/ReadVariableOp2D
 dense_451/BiasAdd/ReadVariableOp dense_451/BiasAdd/ReadVariableOp2B
dense_451/MatMul/ReadVariableOpdense_451/MatMul/ReadVariableOp2D
 dense_452/BiasAdd/ReadVariableOp dense_452/BiasAdd/ReadVariableOp2B
dense_452/MatMul/ReadVariableOpdense_452/MatMul/ReadVariableOp2D
 dense_453/BiasAdd/ReadVariableOp dense_453/BiasAdd/ReadVariableOp2B
dense_453/MatMul/ReadVariableOpdense_453/MatMul/ReadVariableOp2D
 dense_454/BiasAdd/ReadVariableOp dense_454/BiasAdd/ReadVariableOp2B
dense_454/MatMul/ReadVariableOpdense_454/MatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
S__inference_batch_normalization_90_layer_call_and_return_conditional_losses_2140222

inputs/
!batchnorm_readvariableop_resource: 3
%batchnorm_mul_readvariableop_resource: 1
#batchnorm_readvariableop_1_resource: 1
#batchnorm_readvariableop_2_resource: 
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
: P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: ~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:��������� z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
: z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:��������� b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:��������� �
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:��������� : : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
R
$__inference__update_step_xla_2140051
gradient
variable:
��*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*!
_input_shapes
:
��: *
	_noinline(:J F
 
_output_shapes
:
��
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
�

�
F__inference_dense_453_layer_call_and_return_conditional_losses_2140176

inputs0
matmul_readvariableop_resource:@ -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@ *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:��������� Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:��������� w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
��
�
#__inference__traced_restore_2140585
file_prefix5
!assignvariableop_dense_450_kernel:
��0
!assignvariableop_1_dense_450_bias:	�7
#assignvariableop_2_dense_451_kernel:
��0
!assignvariableop_3_dense_451_bias:	�6
#assignvariableop_4_dense_452_kernel:	�@/
!assignvariableop_5_dense_452_bias:@5
#assignvariableop_6_dense_453_kernel:@ /
!assignvariableop_7_dense_453_bias: =
/assignvariableop_8_batch_normalization_90_gamma: <
.assignvariableop_9_batch_normalization_90_beta: D
6assignvariableop_10_batch_normalization_90_moving_mean: H
:assignvariableop_11_batch_normalization_90_moving_variance: 6
$assignvariableop_12_dense_454_kernel: 0
"assignvariableop_13_dense_454_bias:'
assignvariableop_14_iteration:	 +
!assignvariableop_15_learning_rate: ?
+assignvariableop_16_adam_m_dense_450_kernel:
��?
+assignvariableop_17_adam_v_dense_450_kernel:
��8
)assignvariableop_18_adam_m_dense_450_bias:	�8
)assignvariableop_19_adam_v_dense_450_bias:	�?
+assignvariableop_20_adam_m_dense_451_kernel:
��?
+assignvariableop_21_adam_v_dense_451_kernel:
��8
)assignvariableop_22_adam_m_dense_451_bias:	�8
)assignvariableop_23_adam_v_dense_451_bias:	�>
+assignvariableop_24_adam_m_dense_452_kernel:	�@>
+assignvariableop_25_adam_v_dense_452_kernel:	�@7
)assignvariableop_26_adam_m_dense_452_bias:@7
)assignvariableop_27_adam_v_dense_452_bias:@=
+assignvariableop_28_adam_m_dense_453_kernel:@ =
+assignvariableop_29_adam_v_dense_453_kernel:@ 7
)assignvariableop_30_adam_m_dense_453_bias: 7
)assignvariableop_31_adam_v_dense_453_bias: E
7assignvariableop_32_adam_m_batch_normalization_90_gamma: E
7assignvariableop_33_adam_v_batch_normalization_90_gamma: D
6assignvariableop_34_adam_m_batch_normalization_90_beta: D
6assignvariableop_35_adam_v_batch_normalization_90_beta: =
+assignvariableop_36_adam_m_dense_454_kernel: =
+assignvariableop_37_adam_v_dense_454_kernel: 7
)assignvariableop_38_adam_m_dense_454_bias:7
)assignvariableop_39_adam_v_dense_454_bias:%
assignvariableop_40_total_2: %
assignvariableop_41_count_2: %
assignvariableop_42_total_1: %
assignvariableop_43_count_1: #
assignvariableop_44_total: #
assignvariableop_45_count: 
identity_47��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_32�AssignVariableOp_33�AssignVariableOp_34�AssignVariableOp_35�AssignVariableOp_36�AssignVariableOp_37�AssignVariableOp_38�AssignVariableOp_39�AssignVariableOp_4�AssignVariableOp_40�AssignVariableOp_41�AssignVariableOp_42�AssignVariableOp_43�AssignVariableOp_44�AssignVariableOp_45�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:/*
dtype0*�
value�B�/B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:/*
dtype0*q
valuehBf/B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�:::::::::::::::::::::::::::::::::::::::::::::::*=
dtypes3
12/	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOp!assignvariableop_dense_450_kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOp!assignvariableop_1_dense_450_biasIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp#assignvariableop_2_dense_451_kernelIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOp!assignvariableop_3_dense_451_biasIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp#assignvariableop_4_dense_452_kernelIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOp!assignvariableop_5_dense_452_biasIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOp#assignvariableop_6_dense_453_kernelIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOp!assignvariableop_7_dense_453_biasIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOp/assignvariableop_8_batch_normalization_90_gammaIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOp.assignvariableop_9_batch_normalization_90_betaIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOp6assignvariableop_10_batch_normalization_90_moving_meanIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOp:assignvariableop_11_batch_normalization_90_moving_varianceIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOp$assignvariableop_12_dense_454_kernelIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOp"assignvariableop_13_dense_454_biasIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_14AssignVariableOpassignvariableop_14_iterationIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOp!assignvariableop_15_learning_rateIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOp+assignvariableop_16_adam_m_dense_450_kernelIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOp+assignvariableop_17_adam_v_dense_450_kernelIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOp)assignvariableop_18_adam_m_dense_450_biasIdentity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOp)assignvariableop_19_adam_v_dense_450_biasIdentity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_20AssignVariableOp+assignvariableop_20_adam_m_dense_451_kernelIdentity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOp+assignvariableop_21_adam_v_dense_451_kernelIdentity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_22AssignVariableOp)assignvariableop_22_adam_m_dense_451_biasIdentity_22:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_23AssignVariableOp)assignvariableop_23_adam_v_dense_451_biasIdentity_23:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_24AssignVariableOp+assignvariableop_24_adam_m_dense_452_kernelIdentity_24:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_25AssignVariableOp+assignvariableop_25_adam_v_dense_452_kernelIdentity_25:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_26AssignVariableOp)assignvariableop_26_adam_m_dense_452_biasIdentity_26:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_27AssignVariableOp)assignvariableop_27_adam_v_dense_452_biasIdentity_27:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_28AssignVariableOp+assignvariableop_28_adam_m_dense_453_kernelIdentity_28:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_29AssignVariableOp+assignvariableop_29_adam_v_dense_453_kernelIdentity_29:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_30AssignVariableOp)assignvariableop_30_adam_m_dense_453_biasIdentity_30:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_31AssignVariableOp)assignvariableop_31_adam_v_dense_453_biasIdentity_31:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_32AssignVariableOp7assignvariableop_32_adam_m_batch_normalization_90_gammaIdentity_32:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_33AssignVariableOp7assignvariableop_33_adam_v_batch_normalization_90_gammaIdentity_33:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_34AssignVariableOp6assignvariableop_34_adam_m_batch_normalization_90_betaIdentity_34:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_35AssignVariableOp6assignvariableop_35_adam_v_batch_normalization_90_betaIdentity_35:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_36AssignVariableOp+assignvariableop_36_adam_m_dense_454_kernelIdentity_36:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_37AssignVariableOp+assignvariableop_37_adam_v_dense_454_kernelIdentity_37:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_38AssignVariableOp)assignvariableop_38_adam_m_dense_454_biasIdentity_38:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_39AssignVariableOp)assignvariableop_39_adam_v_dense_454_biasIdentity_39:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_40AssignVariableOpassignvariableop_40_total_2Identity_40:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_41AssignVariableOpassignvariableop_41_count_2Identity_41:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_42AssignVariableOpassignvariableop_42_total_1Identity_42:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_43AssignVariableOpassignvariableop_43_count_1Identity_43:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_44AssignVariableOpassignvariableop_44_totalIdentity_44:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_45AssignVariableOpassignvariableop_45_countIdentity_45:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 �
Identity_46Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_47IdentityIdentity_46:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_47Identity_47:output:0*q
_input_shapes`
^: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
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
AssignVariableOp_45AssignVariableOp_452(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
M
$__inference__update_step_xla_2140056
gradient
variable:	�*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes
	:�: *
	_noinline(:E A

_output_shapes	
:�
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
�
P
$__inference__update_step_xla_2140071
gradient
variable:@ *
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes
:@ : *
	_noinline(:H D

_output_shapes

:@ 
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
�
�
+__inference_dense_453_layer_call_fn_2140165

inputs
unknown:@ 
	unknown_0: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *O
fJRH
F__inference_dense_453_layer_call_and_return_conditional_losses_2139482o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:��������� `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�F
�
J__inference_sequential_90_layer_call_and_return_conditional_losses_2139967

inputs<
(dense_450_matmul_readvariableop_resource:
��8
)dense_450_biasadd_readvariableop_resource:	�<
(dense_451_matmul_readvariableop_resource:
��8
)dense_451_biasadd_readvariableop_resource:	�;
(dense_452_matmul_readvariableop_resource:	�@7
)dense_452_biasadd_readvariableop_resource:@:
(dense_453_matmul_readvariableop_resource:@ 7
)dense_453_biasadd_readvariableop_resource: F
8batch_normalization_90_batchnorm_readvariableop_resource: J
<batch_normalization_90_batchnorm_mul_readvariableop_resource: H
:batch_normalization_90_batchnorm_readvariableop_1_resource: H
:batch_normalization_90_batchnorm_readvariableop_2_resource: :
(dense_454_matmul_readvariableop_resource: 7
)dense_454_biasadd_readvariableop_resource:
identity��/batch_normalization_90/batchnorm/ReadVariableOp�1batch_normalization_90/batchnorm/ReadVariableOp_1�1batch_normalization_90/batchnorm/ReadVariableOp_2�3batch_normalization_90/batchnorm/mul/ReadVariableOp� dense_450/BiasAdd/ReadVariableOp�dense_450/MatMul/ReadVariableOp� dense_451/BiasAdd/ReadVariableOp�dense_451/MatMul/ReadVariableOp� dense_452/BiasAdd/ReadVariableOp�dense_452/MatMul/ReadVariableOp� dense_453/BiasAdd/ReadVariableOp�dense_453/MatMul/ReadVariableOp� dense_454/BiasAdd/ReadVariableOp�dense_454/MatMul/ReadVariableOp�
dense_450/MatMul/ReadVariableOpReadVariableOp(dense_450_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0~
dense_450/MatMulMatMulinputs'dense_450/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
 dense_450/BiasAdd/ReadVariableOpReadVariableOp)dense_450_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_450/BiasAddBiasAdddense_450/MatMul:product:0(dense_450/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������e
dense_450/ReluReludense_450/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
dense_451/MatMul/ReadVariableOpReadVariableOp(dense_451_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
dense_451/MatMulMatMuldense_450/Relu:activations:0'dense_451/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
 dense_451/BiasAdd/ReadVariableOpReadVariableOp)dense_451_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_451/BiasAddBiasAdddense_451/MatMul:product:0(dense_451/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������k
dense_451/SigmoidSigmoiddense_451/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
dense_452/MatMul/ReadVariableOpReadVariableOp(dense_452_matmul_readvariableop_resource*
_output_shapes
:	�@*
dtype0�
dense_452/MatMulMatMuldense_451/Sigmoid:y:0'dense_452/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
 dense_452/BiasAdd/ReadVariableOpReadVariableOp)dense_452_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
dense_452/BiasAddBiasAdddense_452/MatMul:product:0(dense_452/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@j
dense_452/SigmoidSigmoiddense_452/BiasAdd:output:0*
T0*'
_output_shapes
:���������@�
dense_453/MatMul/ReadVariableOpReadVariableOp(dense_453_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype0�
dense_453/MatMulMatMuldense_452/Sigmoid:y:0'dense_453/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
 dense_453/BiasAdd/ReadVariableOpReadVariableOp)dense_453_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
dense_453/BiasAddBiasAdddense_453/MatMul:product:0(dense_453/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� j
dense_453/SigmoidSigmoiddense_453/BiasAdd:output:0*
T0*'
_output_shapes
:��������� �
/batch_normalization_90/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_90_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype0k
&batch_normalization_90/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
$batch_normalization_90/batchnorm/addAddV27batch_normalization_90/batchnorm/ReadVariableOp:value:0/batch_normalization_90/batchnorm/add/y:output:0*
T0*
_output_shapes
: ~
&batch_normalization_90/batchnorm/RsqrtRsqrt(batch_normalization_90/batchnorm/add:z:0*
T0*
_output_shapes
: �
3batch_normalization_90/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_90_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype0�
$batch_normalization_90/batchnorm/mulMul*batch_normalization_90/batchnorm/Rsqrt:y:0;batch_normalization_90/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: �
&batch_normalization_90/batchnorm/mul_1Muldense_453/Sigmoid:y:0(batch_normalization_90/batchnorm/mul:z:0*
T0*'
_output_shapes
:��������� �
1batch_normalization_90/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_90_batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype0�
&batch_normalization_90/batchnorm/mul_2Mul9batch_normalization_90/batchnorm/ReadVariableOp_1:value:0(batch_normalization_90/batchnorm/mul:z:0*
T0*
_output_shapes
: �
1batch_normalization_90/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_90_batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype0�
$batch_normalization_90/batchnorm/subSub9batch_normalization_90/batchnorm/ReadVariableOp_2:value:0*batch_normalization_90/batchnorm/mul_2:z:0*
T0*
_output_shapes
: �
&batch_normalization_90/batchnorm/add_1AddV2*batch_normalization_90/batchnorm/mul_1:z:0(batch_normalization_90/batchnorm/sub:z:0*
T0*'
_output_shapes
:��������� �
dense_454/MatMul/ReadVariableOpReadVariableOp(dense_454_matmul_readvariableop_resource*
_output_shapes

: *
dtype0�
dense_454/MatMulMatMul*batch_normalization_90/batchnorm/add_1:z:0'dense_454/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
 dense_454/BiasAdd/ReadVariableOpReadVariableOp)dense_454_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_454/BiasAddBiasAdddense_454/MatMul:product:0(dense_454/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������j
dense_454/SigmoidSigmoiddense_454/BiasAdd:output:0*
T0*'
_output_shapes
:���������d
IdentityIdentitydense_454/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp0^batch_normalization_90/batchnorm/ReadVariableOp2^batch_normalization_90/batchnorm/ReadVariableOp_12^batch_normalization_90/batchnorm/ReadVariableOp_24^batch_normalization_90/batchnorm/mul/ReadVariableOp!^dense_450/BiasAdd/ReadVariableOp ^dense_450/MatMul/ReadVariableOp!^dense_451/BiasAdd/ReadVariableOp ^dense_451/MatMul/ReadVariableOp!^dense_452/BiasAdd/ReadVariableOp ^dense_452/MatMul/ReadVariableOp!^dense_453/BiasAdd/ReadVariableOp ^dense_453/MatMul/ReadVariableOp!^dense_454/BiasAdd/ReadVariableOp ^dense_454/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:����������: : : : : : : : : : : : : : 2b
/batch_normalization_90/batchnorm/ReadVariableOp/batch_normalization_90/batchnorm/ReadVariableOp2f
1batch_normalization_90/batchnorm/ReadVariableOp_11batch_normalization_90/batchnorm/ReadVariableOp_12f
1batch_normalization_90/batchnorm/ReadVariableOp_21batch_normalization_90/batchnorm/ReadVariableOp_22j
3batch_normalization_90/batchnorm/mul/ReadVariableOp3batch_normalization_90/batchnorm/mul/ReadVariableOp2D
 dense_450/BiasAdd/ReadVariableOp dense_450/BiasAdd/ReadVariableOp2B
dense_450/MatMul/ReadVariableOpdense_450/MatMul/ReadVariableOp2D
 dense_451/BiasAdd/ReadVariableOp dense_451/BiasAdd/ReadVariableOp2B
dense_451/MatMul/ReadVariableOpdense_451/MatMul/ReadVariableOp2D
 dense_452/BiasAdd/ReadVariableOp dense_452/BiasAdd/ReadVariableOp2B
dense_452/MatMul/ReadVariableOpdense_452/MatMul/ReadVariableOp2D
 dense_453/BiasAdd/ReadVariableOp dense_453/BiasAdd/ReadVariableOp2B
dense_453/MatMul/ReadVariableOpdense_453/MatMul/ReadVariableOp2D
 dense_454/BiasAdd/ReadVariableOp dense_454/BiasAdd/ReadVariableOp2B
dense_454/MatMul/ReadVariableOpdense_454/MatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�Y
�
 __inference__traced_save_2140437
file_prefix/
+savev2_dense_450_kernel_read_readvariableop-
)savev2_dense_450_bias_read_readvariableop/
+savev2_dense_451_kernel_read_readvariableop-
)savev2_dense_451_bias_read_readvariableop/
+savev2_dense_452_kernel_read_readvariableop-
)savev2_dense_452_bias_read_readvariableop/
+savev2_dense_453_kernel_read_readvariableop-
)savev2_dense_453_bias_read_readvariableop;
7savev2_batch_normalization_90_gamma_read_readvariableop:
6savev2_batch_normalization_90_beta_read_readvariableopA
=savev2_batch_normalization_90_moving_mean_read_readvariableopE
Asavev2_batch_normalization_90_moving_variance_read_readvariableop/
+savev2_dense_454_kernel_read_readvariableop-
)savev2_dense_454_bias_read_readvariableop(
$savev2_iteration_read_readvariableop	,
(savev2_learning_rate_read_readvariableop6
2savev2_adam_m_dense_450_kernel_read_readvariableop6
2savev2_adam_v_dense_450_kernel_read_readvariableop4
0savev2_adam_m_dense_450_bias_read_readvariableop4
0savev2_adam_v_dense_450_bias_read_readvariableop6
2savev2_adam_m_dense_451_kernel_read_readvariableop6
2savev2_adam_v_dense_451_kernel_read_readvariableop4
0savev2_adam_m_dense_451_bias_read_readvariableop4
0savev2_adam_v_dense_451_bias_read_readvariableop6
2savev2_adam_m_dense_452_kernel_read_readvariableop6
2savev2_adam_v_dense_452_kernel_read_readvariableop4
0savev2_adam_m_dense_452_bias_read_readvariableop4
0savev2_adam_v_dense_452_bias_read_readvariableop6
2savev2_adam_m_dense_453_kernel_read_readvariableop6
2savev2_adam_v_dense_453_kernel_read_readvariableop4
0savev2_adam_m_dense_453_bias_read_readvariableop4
0savev2_adam_v_dense_453_bias_read_readvariableopB
>savev2_adam_m_batch_normalization_90_gamma_read_readvariableopB
>savev2_adam_v_batch_normalization_90_gamma_read_readvariableopA
=savev2_adam_m_batch_normalization_90_beta_read_readvariableopA
=savev2_adam_v_batch_normalization_90_beta_read_readvariableop6
2savev2_adam_m_dense_454_kernel_read_readvariableop6
2savev2_adam_v_dense_454_kernel_read_readvariableop4
0savev2_adam_m_dense_454_bias_read_readvariableop4
0savev2_adam_v_dense_454_bias_read_readvariableop&
"savev2_total_2_read_readvariableop&
"savev2_count_2_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop
savev2_const

identity_1��MergeV2Checkpointsw
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
_temp/part�
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
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: �
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:/*
dtype0*�
value�B�/B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:/*
dtype0*q
valuehBf/B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_dense_450_kernel_read_readvariableop)savev2_dense_450_bias_read_readvariableop+savev2_dense_451_kernel_read_readvariableop)savev2_dense_451_bias_read_readvariableop+savev2_dense_452_kernel_read_readvariableop)savev2_dense_452_bias_read_readvariableop+savev2_dense_453_kernel_read_readvariableop)savev2_dense_453_bias_read_readvariableop7savev2_batch_normalization_90_gamma_read_readvariableop6savev2_batch_normalization_90_beta_read_readvariableop=savev2_batch_normalization_90_moving_mean_read_readvariableopAsavev2_batch_normalization_90_moving_variance_read_readvariableop+savev2_dense_454_kernel_read_readvariableop)savev2_dense_454_bias_read_readvariableop$savev2_iteration_read_readvariableop(savev2_learning_rate_read_readvariableop2savev2_adam_m_dense_450_kernel_read_readvariableop2savev2_adam_v_dense_450_kernel_read_readvariableop0savev2_adam_m_dense_450_bias_read_readvariableop0savev2_adam_v_dense_450_bias_read_readvariableop2savev2_adam_m_dense_451_kernel_read_readvariableop2savev2_adam_v_dense_451_kernel_read_readvariableop0savev2_adam_m_dense_451_bias_read_readvariableop0savev2_adam_v_dense_451_bias_read_readvariableop2savev2_adam_m_dense_452_kernel_read_readvariableop2savev2_adam_v_dense_452_kernel_read_readvariableop0savev2_adam_m_dense_452_bias_read_readvariableop0savev2_adam_v_dense_452_bias_read_readvariableop2savev2_adam_m_dense_453_kernel_read_readvariableop2savev2_adam_v_dense_453_kernel_read_readvariableop0savev2_adam_m_dense_453_bias_read_readvariableop0savev2_adam_v_dense_453_bias_read_readvariableop>savev2_adam_m_batch_normalization_90_gamma_read_readvariableop>savev2_adam_v_batch_normalization_90_gamma_read_readvariableop=savev2_adam_m_batch_normalization_90_beta_read_readvariableop=savev2_adam_v_batch_normalization_90_beta_read_readvariableop2savev2_adam_m_dense_454_kernel_read_readvariableop2savev2_adam_v_dense_454_kernel_read_readvariableop0savev2_adam_m_dense_454_bias_read_readvariableop0savev2_adam_v_dense_454_bias_read_readvariableop"savev2_total_2_read_readvariableop"savev2_count_2_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableopsavev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *=
dtypes3
12/	�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
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

identity_1Identity_1:output:0*�
_input_shapes�
�: :
��:�:
��:�:	�@:@:@ : : : : : : :: : :
��:
��:�:�:
��:
��:�:�:	�@:	�@:@:@:@ :@ : : : : : : : : ::: : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:&"
 
_output_shapes
:
��:!

_output_shapes	
:�:&"
 
_output_shapes
:
��:!

_output_shapes	
:�:%!

_output_shapes
:	�@: 

_output_shapes
:@:$ 

_output_shapes

:@ : 

_output_shapes
: : 	

_output_shapes
: : 


_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: :$ 

_output_shapes

: : 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :&"
 
_output_shapes
:
��:&"
 
_output_shapes
:
��:!

_output_shapes	
:�:!

_output_shapes	
:�:&"
 
_output_shapes
:
��:&"
 
_output_shapes
:
��:!

_output_shapes	
:�:!

_output_shapes	
:�:%!

_output_shapes
:	�@:%!

_output_shapes
:	�@: 

_output_shapes
:@: 

_output_shapes
:@:$ 

_output_shapes

:@ :$ 

_output_shapes

:@ : 

_output_shapes
: :  

_output_shapes
: : !

_output_shapes
: : "

_output_shapes
: : #

_output_shapes
: : $

_output_shapes
: :$% 

_output_shapes

: :$& 

_output_shapes

: : '

_output_shapes
:: (

_output_shapes
::)
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
: 
�
Q
$__inference__update_step_xla_2140061
gradient
variable:	�@*
_XlaMustCompile(*(
_construction_contextkEagerRuntime* 
_input_shapes
:	�@: *
	_noinline(:I E

_output_shapes
:	�@
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
�
L
$__inference__update_step_xla_2140086
gradient
variable: *
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

: : *
	_noinline(:D @

_output_shapes
: 
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
�
L
$__inference__update_step_xla_2140076
gradient
variable: *
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

: : *
	_noinline(:D @

_output_shapes
: 
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
�
P
$__inference__update_step_xla_2140091
gradient
variable: *
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes
: : *
	_noinline(:H D

_output_shapes

: 
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
�
M
$__inference__update_step_xla_2140046
gradient
variable:	�*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes
	:�: *
	_noinline(:E A

_output_shapes	
:�
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
�

�
F__inference_dense_454_layer_call_and_return_conditional_losses_2139508

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:���������Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:��������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
8__inference_batch_normalization_90_layer_call_fn_2140202

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *\
fWRU
S__inference_batch_normalization_90_layer_call_and_return_conditional_losses_2139402o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:��������� `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:��������� : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
%__inference_signature_wrapper_2139846
dense_450_input
unknown:
��
	unknown_0:	�
	unknown_1:
��
	unknown_2:	�
	unknown_3:	�@
	unknown_4:@
	unknown_5:@ 
	unknown_6: 
	unknown_7: 
	unknown_8: 
	unknown_9: 

unknown_10: 

unknown_11: 

unknown_12:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCalldense_450_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*0
_read_only_resource_inputs
	
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *+
f&R$
"__inference__wrapped_model_2139331o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:����������: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
(
_output_shapes
:����������
)
_user_specified_namedense_450_input
�
�
+__inference_dense_454_layer_call_fn_2140265

inputs
unknown: 
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *O
fJRH
F__inference_dense_454_layer_call_and_return_conditional_losses_2139508o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:��������� : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�

�
F__inference_dense_452_layer_call_and_return_conditional_losses_2140156

inputs1
matmul_readvariableop_resource:	�@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:���������@Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:���������@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
+__inference_dense_452_layer_call_fn_2140145

inputs
unknown:	�@
	unknown_0:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *O
fJRH
F__inference_dense_452_layer_call_and_return_conditional_losses_2139465o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
/__inference_sequential_90_layer_call_fn_2139879

inputs
unknown:
��
	unknown_0:	�
	unknown_1:
��
	unknown_2:	�
	unknown_3:	�@
	unknown_4:@
	unknown_5:@ 
	unknown_6: 
	unknown_7: 
	unknown_8: 
	unknown_9: 

unknown_10: 

unknown_11: 

unknown_12:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*0
_read_only_resource_inputs
	
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *S
fNRL
J__inference_sequential_90_layer_call_and_return_conditional_losses_2139515o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:����������: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
/__inference_sequential_90_layer_call_fn_2139912

inputs
unknown:
��
	unknown_0:	�
	unknown_1:
��
	unknown_2:	�
	unknown_3:	�@
	unknown_4:@
	unknown_5:@ 
	unknown_6: 
	unknown_7: 
	unknown_8: 
	unknown_9: 

unknown_10: 

unknown_11: 

unknown_12:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*.
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *S
fNRL
J__inference_sequential_90_layer_call_and_return_conditional_losses_2139669o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:����������: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�$
�
J__inference_sequential_90_layer_call_and_return_conditional_losses_2139809
dense_450_input%
dense_450_2139774:
�� 
dense_450_2139776:	�%
dense_451_2139779:
�� 
dense_451_2139781:	�$
dense_452_2139784:	�@
dense_452_2139786:@#
dense_453_2139789:@ 
dense_453_2139791: ,
batch_normalization_90_2139794: ,
batch_normalization_90_2139796: ,
batch_normalization_90_2139798: ,
batch_normalization_90_2139800: #
dense_454_2139803: 
dense_454_2139805:
identity��.batch_normalization_90/StatefulPartitionedCall�!dense_450/StatefulPartitionedCall�!dense_451/StatefulPartitionedCall�!dense_452/StatefulPartitionedCall�!dense_453/StatefulPartitionedCall�!dense_454/StatefulPartitionedCall�
!dense_450/StatefulPartitionedCallStatefulPartitionedCalldense_450_inputdense_450_2139774dense_450_2139776*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *O
fJRH
F__inference_dense_450_layer_call_and_return_conditional_losses_2139431�
!dense_451/StatefulPartitionedCallStatefulPartitionedCall*dense_450/StatefulPartitionedCall:output:0dense_451_2139779dense_451_2139781*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *O
fJRH
F__inference_dense_451_layer_call_and_return_conditional_losses_2139448�
!dense_452/StatefulPartitionedCallStatefulPartitionedCall*dense_451/StatefulPartitionedCall:output:0dense_452_2139784dense_452_2139786*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *O
fJRH
F__inference_dense_452_layer_call_and_return_conditional_losses_2139465�
!dense_453/StatefulPartitionedCallStatefulPartitionedCall*dense_452/StatefulPartitionedCall:output:0dense_453_2139789dense_453_2139791*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *O
fJRH
F__inference_dense_453_layer_call_and_return_conditional_losses_2139482�
.batch_normalization_90/StatefulPartitionedCallStatefulPartitionedCall*dense_453/StatefulPartitionedCall:output:0batch_normalization_90_2139794batch_normalization_90_2139796batch_normalization_90_2139798batch_normalization_90_2139800*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *\
fWRU
S__inference_batch_normalization_90_layer_call_and_return_conditional_losses_2139402�
!dense_454/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_90/StatefulPartitionedCall:output:0dense_454_2139803dense_454_2139805*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *O
fJRH
F__inference_dense_454_layer_call_and_return_conditional_losses_2139508y
IdentityIdentity*dense_454/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp/^batch_normalization_90/StatefulPartitionedCall"^dense_450/StatefulPartitionedCall"^dense_451/StatefulPartitionedCall"^dense_452/StatefulPartitionedCall"^dense_453/StatefulPartitionedCall"^dense_454/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:����������: : : : : : : : : : : : : : 2`
.batch_normalization_90/StatefulPartitionedCall.batch_normalization_90/StatefulPartitionedCall2F
!dense_450/StatefulPartitionedCall!dense_450/StatefulPartitionedCall2F
!dense_451/StatefulPartitionedCall!dense_451/StatefulPartitionedCall2F
!dense_452/StatefulPartitionedCall!dense_452/StatefulPartitionedCall2F
!dense_453/StatefulPartitionedCall!dense_453/StatefulPartitionedCall2F
!dense_454/StatefulPartitionedCall!dense_454/StatefulPartitionedCall:Y U
(
_output_shapes
:����������
)
_user_specified_namedense_450_input"�
L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
L
dense_450_input9
!serving_default_dense_450_input:0����������=
	dense_4540
StatefulPartitionedCall:0���������tensorflow/serving/predict:��
�
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
	variables
trainable_variables
	regularization_losses

	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures"
_tf_keras_sequential
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
�
 	variables
!trainable_variables
"regularization_losses
#	keras_api
$__call__
*%&call_and_return_all_conditional_losses

&kernel
'bias"
_tf_keras_layer
�
(	variables
)trainable_variables
*regularization_losses
+	keras_api
,__call__
*-&call_and_return_all_conditional_losses

.kernel
/bias"
_tf_keras_layer
�
0	variables
1trainable_variables
2regularization_losses
3	keras_api
4__call__
*5&call_and_return_all_conditional_losses
6axis
	7gamma
8beta
9moving_mean
:moving_variance"
_tf_keras_layer
�
;	variables
<trainable_variables
=regularization_losses
>	keras_api
?__call__
*@&call_and_return_all_conditional_losses

Akernel
Bbias"
_tf_keras_layer
�
0
1
2
3
&4
'5
.6
/7
78
89
910
:11
A12
B13"
trackable_list_wrapper
v
0
1
2
3
&4
'5
.6
/7
78
89
A10
B11"
trackable_list_wrapper
 "
trackable_list_wrapper
�
Cnon_trainable_variables

Dlayers
Emetrics
Flayer_regularization_losses
Glayer_metrics
	variables
trainable_variables
	regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
Htrace_0
Itrace_1
Jtrace_2
Ktrace_32�
/__inference_sequential_90_layer_call_fn_2139546
/__inference_sequential_90_layer_call_fn_2139879
/__inference_sequential_90_layer_call_fn_2139912
/__inference_sequential_90_layer_call_fn_2139733�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zHtrace_0zItrace_1zJtrace_2zKtrace_3
�
Ltrace_0
Mtrace_1
Ntrace_2
Otrace_32�
J__inference_sequential_90_layer_call_and_return_conditional_losses_2139967
J__inference_sequential_90_layer_call_and_return_conditional_losses_2140036
J__inference_sequential_90_layer_call_and_return_conditional_losses_2139771
J__inference_sequential_90_layer_call_and_return_conditional_losses_2139809�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zLtrace_0zMtrace_1zNtrace_2zOtrace_3
�B�
"__inference__wrapped_model_2139331dense_450_input"�
���
FullArgSpec
args� 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�
P
_variables
Q_iterations
R_learning_rate
S_index_dict
T
_momentums
U_velocities
V_update_step_xla"
experimentalOptimizer
,
Wserving_default"
signature_map
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
Xnon_trainable_variables

Ylayers
Zmetrics
[layer_regularization_losses
\layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
]trace_02�
+__inference_dense_450_layer_call_fn_2140105�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z]trace_0
�
^trace_02�
F__inference_dense_450_layer_call_and_return_conditional_losses_2140116�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z^trace_0
$:"
��2dense_450/kernel
:�2dense_450/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
_non_trainable_variables

`layers
ametrics
blayer_regularization_losses
clayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
dtrace_02�
+__inference_dense_451_layer_call_fn_2140125�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zdtrace_0
�
etrace_02�
F__inference_dense_451_layer_call_and_return_conditional_losses_2140136�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zetrace_0
$:"
��2dense_451/kernel
:�2dense_451/bias
.
&0
'1"
trackable_list_wrapper
.
&0
'1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
fnon_trainable_variables

glayers
hmetrics
ilayer_regularization_losses
jlayer_metrics
 	variables
!trainable_variables
"regularization_losses
$__call__
*%&call_and_return_all_conditional_losses
&%"call_and_return_conditional_losses"
_generic_user_object
�
ktrace_02�
+__inference_dense_452_layer_call_fn_2140145�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zktrace_0
�
ltrace_02�
F__inference_dense_452_layer_call_and_return_conditional_losses_2140156�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zltrace_0
#:!	�@2dense_452/kernel
:@2dense_452/bias
.
.0
/1"
trackable_list_wrapper
.
.0
/1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
mnon_trainable_variables

nlayers
ometrics
player_regularization_losses
qlayer_metrics
(	variables
)trainable_variables
*regularization_losses
,__call__
*-&call_and_return_all_conditional_losses
&-"call_and_return_conditional_losses"
_generic_user_object
�
rtrace_02�
+__inference_dense_453_layer_call_fn_2140165�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zrtrace_0
�
strace_02�
F__inference_dense_453_layer_call_and_return_conditional_losses_2140176�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zstrace_0
": @ 2dense_453/kernel
: 2dense_453/bias
<
70
81
92
:3"
trackable_list_wrapper
.
70
81"
trackable_list_wrapper
 "
trackable_list_wrapper
�
tnon_trainable_variables

ulayers
vmetrics
wlayer_regularization_losses
xlayer_metrics
0	variables
1trainable_variables
2regularization_losses
4__call__
*5&call_and_return_all_conditional_losses
&5"call_and_return_conditional_losses"
_generic_user_object
�
ytrace_0
ztrace_12�
8__inference_batch_normalization_90_layer_call_fn_2140189
8__inference_batch_normalization_90_layer_call_fn_2140202�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zytrace_0zztrace_1
�
{trace_0
|trace_12�
S__inference_batch_normalization_90_layer_call_and_return_conditional_losses_2140222
S__inference_batch_normalization_90_layer_call_and_return_conditional_losses_2140256�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z{trace_0z|trace_1
 "
trackable_list_wrapper
*:( 2batch_normalization_90/gamma
):' 2batch_normalization_90/beta
2:0  (2"batch_normalization_90/moving_mean
6:4  (2&batch_normalization_90/moving_variance
.
A0
B1"
trackable_list_wrapper
.
A0
B1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
}non_trainable_variables

~layers
metrics
 �layer_regularization_losses
�layer_metrics
;	variables
<trainable_variables
=regularization_losses
?__call__
*@&call_and_return_all_conditional_losses
&@"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
+__inference_dense_454_layer_call_fn_2140265�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
F__inference_dense_454_layer_call_and_return_conditional_losses_2140276�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
":  2dense_454/kernel
:2dense_454/bias
.
90
:1"
trackable_list_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
8
�0
�1
�2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
/__inference_sequential_90_layer_call_fn_2139546dense_450_input"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
/__inference_sequential_90_layer_call_fn_2139879inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
/__inference_sequential_90_layer_call_fn_2139912inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
/__inference_sequential_90_layer_call_fn_2139733dense_450_input"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
J__inference_sequential_90_layer_call_and_return_conditional_losses_2139967inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
J__inference_sequential_90_layer_call_and_return_conditional_losses_2140036inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
J__inference_sequential_90_layer_call_and_return_conditional_losses_2139771dense_450_input"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
J__inference_sequential_90_layer_call_and_return_conditional_losses_2139809dense_450_input"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�
Q0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21
�22
�23
�24"
trackable_list_wrapper
:	 2	iteration
: 2learning_rate
 "
trackable_dict_wrapper
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11"
trackable_list_wrapper
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11"
trackable_list_wrapper
�
�trace_0
�trace_1
�trace_2
�trace_3
�trace_4
�trace_5
�trace_6
�trace_7
�trace_8
�trace_9
�trace_10
�trace_112�
$__inference__update_step_xla_2140041
$__inference__update_step_xla_2140046
$__inference__update_step_xla_2140051
$__inference__update_step_xla_2140056
$__inference__update_step_xla_2140061
$__inference__update_step_xla_2140066
$__inference__update_step_xla_2140071
$__inference__update_step_xla_2140076
$__inference__update_step_xla_2140081
$__inference__update_step_xla_2140086
$__inference__update_step_xla_2140091
$__inference__update_step_xla_2140096�
���
FullArgSpec2
args*�'
jself

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0z�trace_0z�trace_1z�trace_2z�trace_3z�trace_4z�trace_5z�trace_6z�trace_7z�trace_8z�trace_9z�trace_10z�trace_11
�B�
%__inference_signature_wrapper_2139846dense_450_input"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
+__inference_dense_450_layer_call_fn_2140105inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_dense_450_layer_call_and_return_conditional_losses_2140116inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
+__inference_dense_451_layer_call_fn_2140125inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_dense_451_layer_call_and_return_conditional_losses_2140136inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
+__inference_dense_452_layer_call_fn_2140145inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_dense_452_layer_call_and_return_conditional_losses_2140156inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
+__inference_dense_453_layer_call_fn_2140165inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_dense_453_layer_call_and_return_conditional_losses_2140176inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
.
90
:1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
8__inference_batch_normalization_90_layer_call_fn_2140189inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
8__inference_batch_normalization_90_layer_call_fn_2140202inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
S__inference_batch_normalization_90_layer_call_and_return_conditional_losses_2140222inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
S__inference_batch_normalization_90_layer_call_and_return_conditional_losses_2140256inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
+__inference_dense_454_layer_call_fn_2140265inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_dense_454_layer_call_and_return_conditional_losses_2140276inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
R
�	variables
�	keras_api

�total

�count"
_tf_keras_metric
c
�	variables
�	keras_api

�total

�count
�
_fn_kwargs"
_tf_keras_metric
c
�	variables
�	keras_api

�total

�count
�
_fn_kwargs"
_tf_keras_metric
):'
��2Adam/m/dense_450/kernel
):'
��2Adam/v/dense_450/kernel
": �2Adam/m/dense_450/bias
": �2Adam/v/dense_450/bias
):'
��2Adam/m/dense_451/kernel
):'
��2Adam/v/dense_451/kernel
": �2Adam/m/dense_451/bias
": �2Adam/v/dense_451/bias
(:&	�@2Adam/m/dense_452/kernel
(:&	�@2Adam/v/dense_452/kernel
!:@2Adam/m/dense_452/bias
!:@2Adam/v/dense_452/bias
':%@ 2Adam/m/dense_453/kernel
':%@ 2Adam/v/dense_453/kernel
!: 2Adam/m/dense_453/bias
!: 2Adam/v/dense_453/bias
/:- 2#Adam/m/batch_normalization_90/gamma
/:- 2#Adam/v/batch_normalization_90/gamma
.:, 2"Adam/m/batch_normalization_90/beta
.:, 2"Adam/v/batch_normalization_90/beta
':% 2Adam/m/dense_454/kernel
':% 2Adam/v/dense_454/kernel
!:2Adam/m/dense_454/bias
!:2Adam/v/dense_454/bias
�B�
$__inference__update_step_xla_2140041gradientvariable"�
���
FullArgSpec2
args*�'
jself

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
$__inference__update_step_xla_2140046gradientvariable"�
���
FullArgSpec2
args*�'
jself

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
$__inference__update_step_xla_2140051gradientvariable"�
���
FullArgSpec2
args*�'
jself

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
$__inference__update_step_xla_2140056gradientvariable"�
���
FullArgSpec2
args*�'
jself

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
$__inference__update_step_xla_2140061gradientvariable"�
���
FullArgSpec2
args*�'
jself

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
$__inference__update_step_xla_2140066gradientvariable"�
���
FullArgSpec2
args*�'
jself

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
$__inference__update_step_xla_2140071gradientvariable"�
���
FullArgSpec2
args*�'
jself

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
$__inference__update_step_xla_2140076gradientvariable"�
���
FullArgSpec2
args*�'
jself

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
$__inference__update_step_xla_2140081gradientvariable"�
���
FullArgSpec2
args*�'
jself

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
$__inference__update_step_xla_2140086gradientvariable"�
���
FullArgSpec2
args*�'
jself

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
$__inference__update_step_xla_2140091gradientvariable"�
���
FullArgSpec2
args*�'
jself

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
$__inference__update_step_xla_2140096gradientvariable"�
���
FullArgSpec2
args*�'
jself

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper�
$__inference__update_step_xla_2140041rl�i
b�_
�
gradient
��
6�3	�
�
��
�
p
` VariableSpec 
`������?
� "
 �
$__inference__update_step_xla_2140046hb�_
X�U
�
gradient�
1�.	�
��
�
p
` VariableSpec 
`������?
� "
 �
$__inference__update_step_xla_2140051rl�i
b�_
�
gradient
��
6�3	�
�
��
�
p
` VariableSpec 
`�ݐ���?
� "
 �
$__inference__update_step_xla_2140056hb�_
X�U
�
gradient�
1�.	�
��
�
p
` VariableSpec 
`�񐁕�?
� "
 �
$__inference__update_step_xla_2140061pj�g
`�]
�
gradient	�@
5�2	�
�	�@
�
p
` VariableSpec 
`��܄��?
� "
 �
$__inference__update_step_xla_2140066f`�]
V�S
�
gradient@
0�-	�
�@
�
p
` VariableSpec 
`������?
� "
 �
$__inference__update_step_xla_2140071nh�e
^�[
�
gradient@ 
4�1	�
�@ 
�
p
` VariableSpec 
`�Į���?
� "
 �
$__inference__update_step_xla_2140076f`�]
V�S
�
gradient 
0�-	�
� 
�
p
` VariableSpec 
`������?
� "
 �
$__inference__update_step_xla_2140081f`�]
V�S
�
gradient 
0�-	�
� 
�
p
` VariableSpec 
`�˂��?
� "
 �
$__inference__update_step_xla_2140086f`�]
V�S
�
gradient 
0�-	�
� 
�
p
` VariableSpec 
`��˂��?
� "
 �
$__inference__update_step_xla_2140091nh�e
^�[
�
gradient 
4�1	�
� 
�
p
` VariableSpec 
`������?
� "
 �
$__inference__update_step_xla_2140096f`�]
V�S
�
gradient
0�-	�
�
�
p
` VariableSpec 
`������?
� "
 �
"__inference__wrapped_model_2139331�&'./:798AB9�6
/�,
*�'
dense_450_input����������
� "5�2
0
	dense_454#� 
	dense_454����������
S__inference_batch_normalization_90_layer_call_and_return_conditional_losses_2140222i:7983�0
)�&
 �
inputs��������� 
p 
� ",�)
"�
tensor_0��������� 
� �
S__inference_batch_normalization_90_layer_call_and_return_conditional_losses_2140256i9:783�0
)�&
 �
inputs��������� 
p
� ",�)
"�
tensor_0��������� 
� �
8__inference_batch_normalization_90_layer_call_fn_2140189^:7983�0
)�&
 �
inputs��������� 
p 
� "!�
unknown��������� �
8__inference_batch_normalization_90_layer_call_fn_2140202^9:783�0
)�&
 �
inputs��������� 
p
� "!�
unknown��������� �
F__inference_dense_450_layer_call_and_return_conditional_losses_2140116e0�-
&�#
!�
inputs����������
� "-�*
#� 
tensor_0����������
� �
+__inference_dense_450_layer_call_fn_2140105Z0�-
&�#
!�
inputs����������
� ""�
unknown�����������
F__inference_dense_451_layer_call_and_return_conditional_losses_2140136e0�-
&�#
!�
inputs����������
� "-�*
#� 
tensor_0����������
� �
+__inference_dense_451_layer_call_fn_2140125Z0�-
&�#
!�
inputs����������
� ""�
unknown�����������
F__inference_dense_452_layer_call_and_return_conditional_losses_2140156d&'0�-
&�#
!�
inputs����������
� ",�)
"�
tensor_0���������@
� �
+__inference_dense_452_layer_call_fn_2140145Y&'0�-
&�#
!�
inputs����������
� "!�
unknown���������@�
F__inference_dense_453_layer_call_and_return_conditional_losses_2140176c.//�,
%�"
 �
inputs���������@
� ",�)
"�
tensor_0��������� 
� �
+__inference_dense_453_layer_call_fn_2140165X.//�,
%�"
 �
inputs���������@
� "!�
unknown��������� �
F__inference_dense_454_layer_call_and_return_conditional_losses_2140276cAB/�,
%�"
 �
inputs��������� 
� ",�)
"�
tensor_0���������
� �
+__inference_dense_454_layer_call_fn_2140265XAB/�,
%�"
 �
inputs��������� 
� "!�
unknown����������
J__inference_sequential_90_layer_call_and_return_conditional_losses_2139771�&'./:798ABA�>
7�4
*�'
dense_450_input����������
p 

 
� ",�)
"�
tensor_0���������
� �
J__inference_sequential_90_layer_call_and_return_conditional_losses_2139809�&'./9:78ABA�>
7�4
*�'
dense_450_input����������
p

 
� ",�)
"�
tensor_0���������
� �
J__inference_sequential_90_layer_call_and_return_conditional_losses_2139967x&'./:798AB8�5
.�+
!�
inputs����������
p 

 
� ",�)
"�
tensor_0���������
� �
J__inference_sequential_90_layer_call_and_return_conditional_losses_2140036x&'./9:78AB8�5
.�+
!�
inputs����������
p

 
� ",�)
"�
tensor_0���������
� �
/__inference_sequential_90_layer_call_fn_2139546v&'./:798ABA�>
7�4
*�'
dense_450_input����������
p 

 
� "!�
unknown����������
/__inference_sequential_90_layer_call_fn_2139733v&'./9:78ABA�>
7�4
*�'
dense_450_input����������
p

 
� "!�
unknown����������
/__inference_sequential_90_layer_call_fn_2139879m&'./:798AB8�5
.�+
!�
inputs����������
p 

 
� "!�
unknown����������
/__inference_sequential_90_layer_call_fn_2139912m&'./9:78AB8�5
.�+
!�
inputs����������
p

 
� "!�
unknown����������
%__inference_signature_wrapper_2139846�&'./:798ABL�I
� 
B�?
=
dense_450_input*�'
dense_450_input����������"5�2
0
	dense_454#� 
	dense_454���������