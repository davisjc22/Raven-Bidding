??
??
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
?
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
executor_typestring ?
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape?"serve*2.0.02unknown8??
n
	h1/kernelVarHandleOp*
dtype0*
shape
:)*
shared_name	h1/kernel*
_output_shapes
: 
g
h1/kernel/Read/ReadVariableOpReadVariableOp	h1/kernel*
dtype0*
_output_shapes

:)
f
h1/biasVarHandleOp*
shared_name	h1/bias*
_output_shapes
: *
dtype0*
shape:
_
h1/bias/Read/ReadVariableOpReadVariableOph1/bias*
dtype0*
_output_shapes
:
n
	h2/kernelVarHandleOp*
shape
:
*
dtype0*
_output_shapes
: *
shared_name	h2/kernel
g
h2/kernel/Read/ReadVariableOpReadVariableOp	h2/kernel*
dtype0*
_output_shapes

:

f
h2/biasVarHandleOp*
_output_shapes
: *
shape:
*
dtype0*
shared_name	h2/bias
_
h2/bias/Read/ReadVariableOpReadVariableOph2/bias*
_output_shapes
:
*
dtype0
n
	h3/kernelVarHandleOp*
shared_name	h3/kernel*
shape
:
*
dtype0*
_output_shapes
: 
g
h3/kernel/Read/ReadVariableOpReadVariableOp	h3/kernel*
_output_shapes

:
*
dtype0
f
h3/biasVarHandleOp*
_output_shapes
: *
shape:*
shared_name	h3/bias*
dtype0
_
h3/bias/Read/ReadVariableOpReadVariableOph3/bias*
_output_shapes
:*
dtype0
v
output/kernelVarHandleOp*
dtype0*
shared_nameoutput/kernel*
_output_shapes
: *
shape
:
o
!output/kernel/Read/ReadVariableOpReadVariableOpoutput/kernel*
dtype0*
_output_shapes

:
n
output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameoutput/bias
g
output/bias/Read/ReadVariableOpReadVariableOpoutput/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
shared_name	Adam/iter*
shape: *
dtype0	*
_output_shapes
: 
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
shared_nameAdam/beta_1*
dtype0*
_output_shapes
: *
shape: 
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
shared_nameAdam/beta_2*
_output_shapes
: *
shape: *
dtype0
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
shared_name
Adam/decay*
shape: *
_output_shapes
: *
dtype0
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
shape: *#
shared_nameAdam/learning_rate*
_output_shapes
: *
dtype0
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
dtype0*
_output_shapes
: 
^
totalVarHandleOp*
dtype0*
shared_nametotal*
_output_shapes
: *
shape: 
W
total/Read/ReadVariableOpReadVariableOptotal*
dtype0*
_output_shapes
: 
^
countVarHandleOp*
shared_namecount*
shape: *
dtype0*
_output_shapes
: 
W
count/Read/ReadVariableOpReadVariableOpcount*
dtype0*
_output_shapes
: 
|
Adam/h1/kernel/mVarHandleOp*
_output_shapes
: *
shape
:)*!
shared_nameAdam/h1/kernel/m*
dtype0
u
$Adam/h1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/h1/kernel/m*
_output_shapes

:)*
dtype0
t
Adam/h1/bias/mVarHandleOp*
shape:*
dtype0*
shared_nameAdam/h1/bias/m*
_output_shapes
: 
m
"Adam/h1/bias/m/Read/ReadVariableOpReadVariableOpAdam/h1/bias/m*
dtype0*
_output_shapes
:
|
Adam/h2/kernel/mVarHandleOp*!
shared_nameAdam/h2/kernel/m*
shape
:
*
_output_shapes
: *
dtype0
u
$Adam/h2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/h2/kernel/m*
dtype0*
_output_shapes

:

t
Adam/h2/bias/mVarHandleOp*
_output_shapes
: *
shape:
*
dtype0*
shared_nameAdam/h2/bias/m
m
"Adam/h2/bias/m/Read/ReadVariableOpReadVariableOpAdam/h2/bias/m*
dtype0*
_output_shapes
:

|
Adam/h3/kernel/mVarHandleOp*
shape
:
*
dtype0*
_output_shapes
: *!
shared_nameAdam/h3/kernel/m
u
$Adam/h3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/h3/kernel/m*
_output_shapes

:
*
dtype0
t
Adam/h3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shared_nameAdam/h3/bias/m*
shape:
m
"Adam/h3/bias/m/Read/ReadVariableOpReadVariableOpAdam/h3/bias/m*
dtype0*
_output_shapes
:
?
Adam/output/kernel/mVarHandleOp*
dtype0*%
shared_nameAdam/output/kernel/m*
shape
:*
_output_shapes
: 
}
(Adam/output/kernel/m/Read/ReadVariableOpReadVariableOpAdam/output/kernel/m*
dtype0*
_output_shapes

:
|
Adam/output/bias/mVarHandleOp*
_output_shapes
: *
shape:*#
shared_nameAdam/output/bias/m*
dtype0
u
&Adam/output/bias/m/Read/ReadVariableOpReadVariableOpAdam/output/bias/m*
_output_shapes
:*
dtype0
|
Adam/h1/kernel/vVarHandleOp*!
shared_nameAdam/h1/kernel/v*
shape
:)*
dtype0*
_output_shapes
: 
u
$Adam/h1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/h1/kernel/v*
_output_shapes

:)*
dtype0
t
Adam/h1/bias/vVarHandleOp*
shape:*
shared_nameAdam/h1/bias/v*
dtype0*
_output_shapes
: 
m
"Adam/h1/bias/v/Read/ReadVariableOpReadVariableOpAdam/h1/bias/v*
dtype0*
_output_shapes
:
|
Adam/h2/kernel/vVarHandleOp*
_output_shapes
: *
shape
:
*
dtype0*!
shared_nameAdam/h2/kernel/v
u
$Adam/h2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/h2/kernel/v*
dtype0*
_output_shapes

:

t
Adam/h2/bias/vVarHandleOp*
shape:
*
shared_nameAdam/h2/bias/v*
_output_shapes
: *
dtype0
m
"Adam/h2/bias/v/Read/ReadVariableOpReadVariableOpAdam/h2/bias/v*
_output_shapes
:
*
dtype0
|
Adam/h3/kernel/vVarHandleOp*!
shared_nameAdam/h3/kernel/v*
dtype0*
shape
:
*
_output_shapes
: 
u
$Adam/h3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/h3/kernel/v*
_output_shapes

:
*
dtype0
t
Adam/h3/bias/vVarHandleOp*
_output_shapes
: *
shape:*
shared_nameAdam/h3/bias/v*
dtype0
m
"Adam/h3/bias/v/Read/ReadVariableOpReadVariableOpAdam/h3/bias/v*
dtype0*
_output_shapes
:
?
Adam/output/kernel/vVarHandleOp*
dtype0*
_output_shapes
: *
shape
:*%
shared_nameAdam/output/kernel/v
}
(Adam/output/kernel/v/Read/ReadVariableOpReadVariableOpAdam/output/kernel/v*
_output_shapes

:*
dtype0
|
Adam/output/bias/vVarHandleOp*
_output_shapes
: *
dtype0*#
shared_nameAdam/output/bias/v*
shape:
u
&Adam/output/bias/v/Read/ReadVariableOpReadVariableOpAdam/output/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
?4
ConstConst"/device:CPU:0*
dtype0*?3
value?3B?3 B?3
?
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer_with_weights-2
layer-5
layer-6
layer_with_weights-3
layer-7
		optimizer

trainable_variables
	variables
regularization_losses
	keras_api

signatures
R
trainable_variables
	variables
regularization_losses
	keras_api
h

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
R
trainable_variables
	variables
regularization_losses
	keras_api
h

kernel
bias
trainable_variables
 	variables
!regularization_losses
"	keras_api
R
#trainable_variables
$	variables
%regularization_losses
&	keras_api
h

'kernel
(bias
)trainable_variables
*	variables
+regularization_losses
,	keras_api
R
-trainable_variables
.	variables
/regularization_losses
0	keras_api
h

1kernel
2bias
3trainable_variables
4	variables
5regularization_losses
6	keras_api
?
7iter

8beta_1

9beta_2
	:decay
;learning_ratemlmmmnmo'mp(mq1mr2msvtvuvvvw'vx(vy1vz2v{
8
0
1
2
3
'4
(5
16
27
8
0
1
2
3
'4
(5
16
27
 
?
<metrics

trainable_variables
	variables
=layer_regularization_losses
regularization_losses
>non_trainable_variables

?layers
 
 
 
 
?
@metrics
trainable_variables
	variables
Alayer_regularization_losses
regularization_losses
Bnon_trainable_variables

Clayers
US
VARIABLE_VALUE	h1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEh1/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
Dmetrics
trainable_variables
	variables
Elayer_regularization_losses
regularization_losses
Fnon_trainable_variables

Glayers
 
 
 
?
Hmetrics
trainable_variables
	variables
Ilayer_regularization_losses
regularization_losses
Jnon_trainable_variables

Klayers
US
VARIABLE_VALUE	h2/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEh2/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
Lmetrics
trainable_variables
 	variables
Mlayer_regularization_losses
!regularization_losses
Nnon_trainable_variables

Olayers
 
 
 
?
Pmetrics
#trainable_variables
$	variables
Qlayer_regularization_losses
%regularization_losses
Rnon_trainable_variables

Slayers
US
VARIABLE_VALUE	h3/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEh3/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

'0
(1

'0
(1
 
?
Tmetrics
)trainable_variables
*	variables
Ulayer_regularization_losses
+regularization_losses
Vnon_trainable_variables

Wlayers
 
 
 
?
Xmetrics
-trainable_variables
.	variables
Ylayer_regularization_losses
/regularization_losses
Znon_trainable_variables

[layers
YW
VARIABLE_VALUEoutput/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEoutput/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

10
21

10
21
 
?
\metrics
3trainable_variables
4	variables
]layer_regularization_losses
5regularization_losses
^non_trainable_variables

_layers
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE

`0
 
 
1
0
1
2
3
4
5
6
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
x
	atotal
	bcount
c
_fn_kwargs
dtrainable_variables
e	variables
fregularization_losses
g	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE
 
 

a0
b1
 
?
hmetrics
dtrainable_variables
e	variables
ilayer_regularization_losses
fregularization_losses
jnon_trainable_variables

klayers
 
 

a0
b1
 
xv
VARIABLE_VALUEAdam/h1/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUEAdam/h1/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/h2/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUEAdam/h2/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/h3/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUEAdam/h3/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/output/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/output/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/h1/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUEAdam/h1/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/h2/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUEAdam/h2/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/h3/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUEAdam/h3/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/output/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/output/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
_output_shapes
: 
{
serving_default_h1_inputPlaceholder*'
_output_shapes
:?????????)*
shape:?????????)*
dtype0
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_h1_input	h1/kernelh1/bias	h2/kernelh2/bias	h3/kernelh3/biasoutput/kerneloutput/bias*-
_gradient_op_typePartitionedCall-115886**
config_proto

CPU

GPU 2J 8*-
f(R&
$__inference_signature_wrapper_115500*'
_output_shapes
:?????????*
Tout
2*
Tin
2	
O
saver_filenamePlaceholder*
shape: *
dtype0*
_output_shapes
: 
?

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameh1/kernel/Read/ReadVariableOph1/bias/Read/ReadVariableOph2/kernel/Read/ReadVariableOph2/bias/Read/ReadVariableOph3/kernel/Read/ReadVariableOph3/bias/Read/ReadVariableOp!output/kernel/Read/ReadVariableOpoutput/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp$Adam/h1/kernel/m/Read/ReadVariableOp"Adam/h1/bias/m/Read/ReadVariableOp$Adam/h2/kernel/m/Read/ReadVariableOp"Adam/h2/bias/m/Read/ReadVariableOp$Adam/h3/kernel/m/Read/ReadVariableOp"Adam/h3/bias/m/Read/ReadVariableOp(Adam/output/kernel/m/Read/ReadVariableOp&Adam/output/bias/m/Read/ReadVariableOp$Adam/h1/kernel/v/Read/ReadVariableOp"Adam/h1/bias/v/Read/ReadVariableOp$Adam/h2/kernel/v/Read/ReadVariableOp"Adam/h2/bias/v/Read/ReadVariableOp$Adam/h3/kernel/v/Read/ReadVariableOp"Adam/h3/bias/v/Read/ReadVariableOp(Adam/output/kernel/v/Read/ReadVariableOp&Adam/output/bias/v/Read/ReadVariableOpConst*-
_gradient_op_typePartitionedCall-115939*
Tout
2*(
f#R!
__inference__traced_save_115938*
_output_shapes
: *,
Tin%
#2!	**
config_proto

CPU

GPU 2J 8
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename	h1/kernelh1/bias	h2/kernelh2/bias	h3/kernelh3/biasoutput/kerneloutput/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcountAdam/h1/kernel/mAdam/h1/bias/mAdam/h2/kernel/mAdam/h2/bias/mAdam/h3/kernel/mAdam/h3/bias/mAdam/output/kernel/mAdam/output/bias/mAdam/h1/kernel/vAdam/h1/bias/vAdam/h2/kernel/vAdam/h2/bias/vAdam/h3/kernel/vAdam/h3/bias/vAdam/output/kernel/vAdam/output/bias/v*
Tout
2*+
f&R$
"__inference__traced_restore_116044**
config_proto

CPU

GPU 2J 8*-
_gradient_op_typePartitionedCall-116045*
_output_shapes
: *+
Tin$
"2 ??
?
?
#__inference_h2_layer_call_fn_115714

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2**
config_proto

CPU

GPU 2J 8*'
_output_shapes
:?????????
*G
fBR@
>__inference_h2_layer_call_and_return_conditional_losses_115229*-
_gradient_op_typePartitionedCall-115235*
Tin
2*
Tout
2?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
"
identityIdentity:output:0*.
_input_shapes
:?????????::22
StatefulPartitionedCallStatefulPartitionedCall: : :& "
 
_user_specified_nameinputs
?

?
'__inference_DeepNN_layer_call_fn_115630

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8*'
_output_shapes
:?????????*
Tout
2*-
_gradient_op_typePartitionedCall-115435*
Tin
2	*K
fFRD
B__inference_DeepNN_layer_call_and_return_conditional_losses_115434**
config_proto

CPU

GPU 2J 8?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*'
_output_shapes
:?????????*
T0"
identityIdentity:output:0*F
_input_shapes5
3:?????????)::::::::22
StatefulPartitionedCallStatefulPartitionedCall: : : : : : : :& "
 
_user_specified_nameinputs: 
?v
?
"__inference__traced_restore_116044
file_prefix
assignvariableop_h1_kernel
assignvariableop_1_h1_bias 
assignvariableop_2_h2_kernel
assignvariableop_3_h2_bias 
assignvariableop_4_h3_kernel
assignvariableop_5_h3_bias$
 assignvariableop_6_output_kernel"
assignvariableop_7_output_bias 
assignvariableop_8_adam_iter"
assignvariableop_9_adam_beta_1#
assignvariableop_10_adam_beta_2"
assignvariableop_11_adam_decay*
&assignvariableop_12_adam_learning_rate
assignvariableop_13_total
assignvariableop_14_count(
$assignvariableop_15_adam_h1_kernel_m&
"assignvariableop_16_adam_h1_bias_m(
$assignvariableop_17_adam_h2_kernel_m&
"assignvariableop_18_adam_h2_bias_m(
$assignvariableop_19_adam_h3_kernel_m&
"assignvariableop_20_adam_h3_bias_m,
(assignvariableop_21_adam_output_kernel_m*
&assignvariableop_22_adam_output_bias_m(
$assignvariableop_23_adam_h1_kernel_v&
"assignvariableop_24_adam_h1_bias_v(
$assignvariableop_25_adam_h2_kernel_v&
"assignvariableop_26_adam_h2_bias_v(
$assignvariableop_27_adam_h3_kernel_v&
"assignvariableop_28_adam_h3_bias_v,
(assignvariableop_29_adam_output_kernel_v*
&assignvariableop_30_adam_output_bias_v
identity_32??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?	RestoreV2?RestoreV2_1?
RestoreV2/tensor_namesConst"/device:CPU:0*?
value?B?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Q
valueHBFB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes~
|:::::::::::::::::::::::::::::::*-
dtypes#
!2	L
IdentityIdentityRestoreV2:tensors:0*
T0*
_output_shapes
:v
AssignVariableOpAssignVariableOpassignvariableop_h1_kernelIdentity:output:0*
dtype0*
_output_shapes
 N

Identity_1IdentityRestoreV2:tensors:1*
_output_shapes
:*
T0z
AssignVariableOp_1AssignVariableOpassignvariableop_1_h1_biasIdentity_1:output:0*
_output_shapes
 *
dtype0N

Identity_2IdentityRestoreV2:tensors:2*
_output_shapes
:*
T0|
AssignVariableOp_2AssignVariableOpassignvariableop_2_h2_kernelIdentity_2:output:0*
dtype0*
_output_shapes
 N

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:z
AssignVariableOp_3AssignVariableOpassignvariableop_3_h2_biasIdentity_3:output:0*
_output_shapes
 *
dtype0N

Identity_4IdentityRestoreV2:tensors:4*
_output_shapes
:*
T0|
AssignVariableOp_4AssignVariableOpassignvariableop_4_h3_kernelIdentity_4:output:0*
dtype0*
_output_shapes
 N

Identity_5IdentityRestoreV2:tensors:5*
T0*
_output_shapes
:z
AssignVariableOp_5AssignVariableOpassignvariableop_5_h3_biasIdentity_5:output:0*
_output_shapes
 *
dtype0N

Identity_6IdentityRestoreV2:tensors:6*
_output_shapes
:*
T0?
AssignVariableOp_6AssignVariableOp assignvariableop_6_output_kernelIdentity_6:output:0*
dtype0*
_output_shapes
 N

Identity_7IdentityRestoreV2:tensors:7*
T0*
_output_shapes
:~
AssignVariableOp_7AssignVariableOpassignvariableop_7_output_biasIdentity_7:output:0*
_output_shapes
 *
dtype0N

Identity_8IdentityRestoreV2:tensors:8*
_output_shapes
:*
T0	|
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_iterIdentity_8:output:0*
dtype0	*
_output_shapes
 N

Identity_9IdentityRestoreV2:tensors:9*
T0*
_output_shapes
:~
AssignVariableOp_9AssignVariableOpassignvariableop_9_adam_beta_1Identity_9:output:0*
_output_shapes
 *
dtype0P
Identity_10IdentityRestoreV2:tensors:10*
T0*
_output_shapes
:?
AssignVariableOp_10AssignVariableOpassignvariableop_10_adam_beta_2Identity_10:output:0*
_output_shapes
 *
dtype0P
Identity_11IdentityRestoreV2:tensors:11*
T0*
_output_shapes
:?
AssignVariableOp_11AssignVariableOpassignvariableop_11_adam_decayIdentity_11:output:0*
dtype0*
_output_shapes
 P
Identity_12IdentityRestoreV2:tensors:12*
_output_shapes
:*
T0?
AssignVariableOp_12AssignVariableOp&assignvariableop_12_adam_learning_rateIdentity_12:output:0*
dtype0*
_output_shapes
 P
Identity_13IdentityRestoreV2:tensors:13*
T0*
_output_shapes
:{
AssignVariableOp_13AssignVariableOpassignvariableop_13_totalIdentity_13:output:0*
dtype0*
_output_shapes
 P
Identity_14IdentityRestoreV2:tensors:14*
_output_shapes
:*
T0{
AssignVariableOp_14AssignVariableOpassignvariableop_14_countIdentity_14:output:0*
dtype0*
_output_shapes
 P
Identity_15IdentityRestoreV2:tensors:15*
T0*
_output_shapes
:?
AssignVariableOp_15AssignVariableOp$assignvariableop_15_adam_h1_kernel_mIdentity_15:output:0*
dtype0*
_output_shapes
 P
Identity_16IdentityRestoreV2:tensors:16*
T0*
_output_shapes
:?
AssignVariableOp_16AssignVariableOp"assignvariableop_16_adam_h1_bias_mIdentity_16:output:0*
_output_shapes
 *
dtype0P
Identity_17IdentityRestoreV2:tensors:17*
T0*
_output_shapes
:?
AssignVariableOp_17AssignVariableOp$assignvariableop_17_adam_h2_kernel_mIdentity_17:output:0*
_output_shapes
 *
dtype0P
Identity_18IdentityRestoreV2:tensors:18*
T0*
_output_shapes
:?
AssignVariableOp_18AssignVariableOp"assignvariableop_18_adam_h2_bias_mIdentity_18:output:0*
_output_shapes
 *
dtype0P
Identity_19IdentityRestoreV2:tensors:19*
_output_shapes
:*
T0?
AssignVariableOp_19AssignVariableOp$assignvariableop_19_adam_h3_kernel_mIdentity_19:output:0*
_output_shapes
 *
dtype0P
Identity_20IdentityRestoreV2:tensors:20*
_output_shapes
:*
T0?
AssignVariableOp_20AssignVariableOp"assignvariableop_20_adam_h3_bias_mIdentity_20:output:0*
_output_shapes
 *
dtype0P
Identity_21IdentityRestoreV2:tensors:21*
T0*
_output_shapes
:?
AssignVariableOp_21AssignVariableOp(assignvariableop_21_adam_output_kernel_mIdentity_21:output:0*
_output_shapes
 *
dtype0P
Identity_22IdentityRestoreV2:tensors:22*
_output_shapes
:*
T0?
AssignVariableOp_22AssignVariableOp&assignvariableop_22_adam_output_bias_mIdentity_22:output:0*
_output_shapes
 *
dtype0P
Identity_23IdentityRestoreV2:tensors:23*
T0*
_output_shapes
:?
AssignVariableOp_23AssignVariableOp$assignvariableop_23_adam_h1_kernel_vIdentity_23:output:0*
dtype0*
_output_shapes
 P
Identity_24IdentityRestoreV2:tensors:24*
T0*
_output_shapes
:?
AssignVariableOp_24AssignVariableOp"assignvariableop_24_adam_h1_bias_vIdentity_24:output:0*
dtype0*
_output_shapes
 P
Identity_25IdentityRestoreV2:tensors:25*
_output_shapes
:*
T0?
AssignVariableOp_25AssignVariableOp$assignvariableop_25_adam_h2_kernel_vIdentity_25:output:0*
_output_shapes
 *
dtype0P
Identity_26IdentityRestoreV2:tensors:26*
T0*
_output_shapes
:?
AssignVariableOp_26AssignVariableOp"assignvariableop_26_adam_h2_bias_vIdentity_26:output:0*
dtype0*
_output_shapes
 P
Identity_27IdentityRestoreV2:tensors:27*
T0*
_output_shapes
:?
AssignVariableOp_27AssignVariableOp$assignvariableop_27_adam_h3_kernel_vIdentity_27:output:0*
dtype0*
_output_shapes
 P
Identity_28IdentityRestoreV2:tensors:28*
T0*
_output_shapes
:?
AssignVariableOp_28AssignVariableOp"assignvariableop_28_adam_h3_bias_vIdentity_28:output:0*
_output_shapes
 *
dtype0P
Identity_29IdentityRestoreV2:tensors:29*
_output_shapes
:*
T0?
AssignVariableOp_29AssignVariableOp(assignvariableop_29_adam_output_kernel_vIdentity_29:output:0*
dtype0*
_output_shapes
 P
Identity_30IdentityRestoreV2:tensors:30*
T0*
_output_shapes
:?
AssignVariableOp_30AssignVariableOp&assignvariableop_30_adam_output_bias_vIdentity_30:output:0*
dtype0*
_output_shapes
 ?
RestoreV2_1/tensor_namesConst"/device:CPU:0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH*
_output_shapes
:*
dtype0t
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
valueB
B *
_output_shapes
:*
dtype0?
RestoreV2_1	RestoreV2file_prefix!RestoreV2_1/tensor_names:output:0%RestoreV2_1/shape_and_slices:output:0
^RestoreV2"/device:CPU:0*
dtypes
2*
_output_shapes
:1
NoOpNoOp"/device:CPU:0*
_output_shapes
 ?
Identity_31Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: ?
Identity_32IdentityIdentity_31:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9
^RestoreV2^RestoreV2_1*
_output_shapes
: *
T0"#
identity_32Identity_32:output:0*?
_input_shapes?
~: :::::::::::::::::::::::::::::::2*
AssignVariableOp_19AssignVariableOp_192*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272$
AssignVariableOpAssignVariableOp2*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92
	RestoreV2	RestoreV22*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112
RestoreV2_1RestoreV2_12*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_23AssignVariableOp_23: : : : : : : : : :+ '
%
_user_specified_namefile_prefix: : : : : : : : :	 :
 : : : : : : : : : : : : 
?	
?
>__inference_h1_layer_call_and_return_conditional_losses_115654

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:)i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:??????????
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*.
_input_shapes
:?????????)::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp: : :& "
 
_user_specified_nameinputs
?	
?
>__inference_h3_layer_call_and_return_conditional_losses_115301

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes

:
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*'
_output_shapes
:?????????*
T0?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:??????????
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*'
_output_shapes
:?????????*
T0"
identityIdentity:output:0*.
_input_shapes
:?????????
::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp: :& "
 
_user_specified_nameinputs: 
?
?
#__inference_h1_layer_call_fn_115661

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*'
_output_shapes
:?????????*G
fBR@
>__inference_h1_layer_call_and_return_conditional_losses_115157*-
_gradient_op_typePartitionedCall-115163*
Tin
2**
config_proto

CPU

GPU 2J 8*
Tout
2?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*.
_input_shapes
:?????????)::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : 
?
`
A__inference_drop2_layer_call_and_return_conditional_losses_115734

inputs
identity?Q
dropout/rateConst*
_output_shapes
: *
dtype0*
valueB
 *??L>C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:_
dropout/random_uniform/minConst*
valueB
 *    *
dtype0*
_output_shapes
: _
dropout/random_uniform/maxConst*
valueB
 *  ??*
dtype0*
_output_shapes
: ?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:?????????
*
dtype0?
dropout/random_uniform/subSub#dropout/random_uniform/max:output:0#dropout/random_uniform/min:output:0*
_output_shapes
: *
T0?
dropout/random_uniform/mulMul-dropout/random_uniform/RandomUniform:output:0dropout/random_uniform/sub:z:0*
T0*'
_output_shapes
:?????????
?
dropout/random_uniformAdddropout/random_uniform/mul:z:0#dropout/random_uniform/min:output:0*
T0*'
_output_shapes
:?????????
R
dropout/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??b
dropout/subSubdropout/sub/x:output:0dropout/rate:output:0*
_output_shapes
: *
T0V
dropout/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??h
dropout/truedivRealDivdropout/truediv/x:output:0dropout/sub:z:0*
_output_shapes
: *
T0?
dropout/GreaterEqualGreaterEqualdropout/random_uniform:z:0dropout/rate:output:0*'
_output_shapes
:?????????
*
T0a
dropout/mulMulinputsdropout/truediv:z:0*'
_output_shapes
:?????????
*
T0o
dropout/CastCastdropout/GreaterEqual:z:0*

SrcT0
*

DstT0*'
_output_shapes
:?????????
i
dropout/mul_1Muldropout/mul:z:0dropout/Cast:y:0*'
_output_shapes
:?????????
*
T0Y
IdentityIdentitydropout/mul_1:z:0*
T0*'
_output_shapes
:?????????
"
identityIdentity:output:0*&
_input_shapes
:?????????
:& "
 
_user_specified_nameinputs
?!
?
B__inference_DeepNN_layer_call_and_return_conditional_losses_115434

inputs%
!h1_statefulpartitionedcall_args_1%
!h1_statefulpartitionedcall_args_2%
!h2_statefulpartitionedcall_args_1%
!h2_statefulpartitionedcall_args_2%
!h3_statefulpartitionedcall_args_1%
!h3_statefulpartitionedcall_args_2)
%output_statefulpartitionedcall_args_1)
%output_statefulpartitionedcall_args_2
identity??drop1/StatefulPartitionedCall?drop2/StatefulPartitionedCall?drop3/StatefulPartitionedCall?h1/StatefulPartitionedCall?h2/StatefulPartitionedCall?h3/StatefulPartitionedCall?output/StatefulPartitionedCall?
h1/StatefulPartitionedCallStatefulPartitionedCallinputs!h1_statefulpartitionedcall_args_1!h1_statefulpartitionedcall_args_2*'
_output_shapes
:?????????*G
fBR@
>__inference_h1_layer_call_and_return_conditional_losses_115157*-
_gradient_op_typePartitionedCall-115163*
Tin
2**
config_proto

CPU

GPU 2J 8*
Tout
2?
drop1/StatefulPartitionedCallStatefulPartitionedCall#h1/StatefulPartitionedCall:output:0*-
_gradient_op_typePartitionedCall-115205*J
fERC
A__inference_drop1_layer_call_and_return_conditional_losses_115194*
Tin
2*'
_output_shapes
:?????????*
Tout
2**
config_proto

CPU

GPU 2J 8?
h2/StatefulPartitionedCallStatefulPartitionedCall&drop1/StatefulPartitionedCall:output:0!h2_statefulpartitionedcall_args_1!h2_statefulpartitionedcall_args_2*G
fBR@
>__inference_h2_layer_call_and_return_conditional_losses_115229*-
_gradient_op_typePartitionedCall-115235*
Tin
2**
config_proto

CPU

GPU 2J 8*
Tout
2*'
_output_shapes
:?????????
?
drop2/StatefulPartitionedCallStatefulPartitionedCall#h2/StatefulPartitionedCall:output:0^drop1/StatefulPartitionedCall*
Tin
2*
Tout
2*'
_output_shapes
:?????????
**
config_proto

CPU

GPU 2J 8*-
_gradient_op_typePartitionedCall-115277*J
fERC
A__inference_drop2_layer_call_and_return_conditional_losses_115266?
h3/StatefulPartitionedCallStatefulPartitionedCall&drop2/StatefulPartitionedCall:output:0!h3_statefulpartitionedcall_args_1!h3_statefulpartitionedcall_args_2*G
fBR@
>__inference_h3_layer_call_and_return_conditional_losses_115301*
Tin
2*'
_output_shapes
:?????????*
Tout
2*-
_gradient_op_typePartitionedCall-115307**
config_proto

CPU

GPU 2J 8?
drop3/StatefulPartitionedCallStatefulPartitionedCall#h3/StatefulPartitionedCall:output:0^drop2/StatefulPartitionedCall*-
_gradient_op_typePartitionedCall-115349*'
_output_shapes
:?????????*
Tin
2**
config_proto

CPU

GPU 2J 8*J
fERC
A__inference_drop3_layer_call_and_return_conditional_losses_115338*
Tout
2?
output/StatefulPartitionedCallStatefulPartitionedCall&drop3/StatefulPartitionedCall:output:0%output_statefulpartitionedcall_args_1%output_statefulpartitionedcall_args_2**
config_proto

CPU

GPU 2J 8*'
_output_shapes
:?????????*K
fFRD
B__inference_output_layer_call_and_return_conditional_losses_115373*
Tin
2*
Tout
2*-
_gradient_op_typePartitionedCall-115379?
IdentityIdentity'output/StatefulPartitionedCall:output:0^drop1/StatefulPartitionedCall^drop2/StatefulPartitionedCall^drop3/StatefulPartitionedCall^h1/StatefulPartitionedCall^h2/StatefulPartitionedCall^h3/StatefulPartitionedCall^output/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*F
_input_shapes5
3:?????????)::::::::28
h1/StatefulPartitionedCallh1/StatefulPartitionedCall28
h2/StatefulPartitionedCallh2/StatefulPartitionedCall28
h3/StatefulPartitionedCallh3/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall2>
drop1/StatefulPartitionedCalldrop1/StatefulPartitionedCall2>
drop2/StatefulPartitionedCalldrop2/StatefulPartitionedCall2>
drop3/StatefulPartitionedCalldrop3/StatefulPartitionedCall:& "
 
_user_specified_nameinputs: : : : : : : : 
?

?
'__inference_DeepNN_layer_call_fn_115643

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8*-
_gradient_op_typePartitionedCall-115470*K
fFRD
B__inference_DeepNN_layer_call_and_return_conditional_losses_115469*
Tout
2*'
_output_shapes
:?????????**
config_proto

CPU

GPU 2J 8*
Tin
2	?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*'
_output_shapes
:?????????*
T0"
identityIdentity:output:0*F
_input_shapes5
3:?????????)::::::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : : : : : : : 
?
_
A__inference_drop1_layer_call_and_return_conditional_losses_115201

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:?????????[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:?????????"!

identity_1Identity_1:output:0*&
_input_shapes
:?????????:& "
 
_user_specified_nameinputs
?
?
B__inference_DeepNN_layer_call_and_return_conditional_losses_115469

inputs%
!h1_statefulpartitionedcall_args_1%
!h1_statefulpartitionedcall_args_2%
!h2_statefulpartitionedcall_args_1%
!h2_statefulpartitionedcall_args_2%
!h3_statefulpartitionedcall_args_1%
!h3_statefulpartitionedcall_args_2)
%output_statefulpartitionedcall_args_1)
%output_statefulpartitionedcall_args_2
identity??h1/StatefulPartitionedCall?h2/StatefulPartitionedCall?h3/StatefulPartitionedCall?output/StatefulPartitionedCall?
h1/StatefulPartitionedCallStatefulPartitionedCallinputs!h1_statefulpartitionedcall_args_1!h1_statefulpartitionedcall_args_2*
Tout
2*'
_output_shapes
:?????????*G
fBR@
>__inference_h1_layer_call_and_return_conditional_losses_115157*-
_gradient_op_typePartitionedCall-115163*
Tin
2**
config_proto

CPU

GPU 2J 8?
drop1/PartitionedCallPartitionedCall#h1/StatefulPartitionedCall:output:0*
Tin
2**
config_proto

CPU

GPU 2J 8*-
_gradient_op_typePartitionedCall-115213*'
_output_shapes
:?????????*
Tout
2*J
fERC
A__inference_drop1_layer_call_and_return_conditional_losses_115201?
h2/StatefulPartitionedCallStatefulPartitionedCalldrop1/PartitionedCall:output:0!h2_statefulpartitionedcall_args_1!h2_statefulpartitionedcall_args_2*
Tout
2**
config_proto

CPU

GPU 2J 8*G
fBR@
>__inference_h2_layer_call_and_return_conditional_losses_115229*-
_gradient_op_typePartitionedCall-115235*
Tin
2*'
_output_shapes
:?????????
?
drop2/PartitionedCallPartitionedCall#h2/StatefulPartitionedCall:output:0*
Tout
2*J
fERC
A__inference_drop2_layer_call_and_return_conditional_losses_115273*'
_output_shapes
:?????????
*-
_gradient_op_typePartitionedCall-115285**
config_proto

CPU

GPU 2J 8*
Tin
2?
h3/StatefulPartitionedCallStatefulPartitionedCalldrop2/PartitionedCall:output:0!h3_statefulpartitionedcall_args_1!h3_statefulpartitionedcall_args_2*'
_output_shapes
:?????????*
Tout
2*G
fBR@
>__inference_h3_layer_call_and_return_conditional_losses_115301*-
_gradient_op_typePartitionedCall-115307*
Tin
2**
config_proto

CPU

GPU 2J 8?
drop3/PartitionedCallPartitionedCall#h3/StatefulPartitionedCall:output:0*J
fERC
A__inference_drop3_layer_call_and_return_conditional_losses_115345*
Tin
2**
config_proto

CPU

GPU 2J 8*-
_gradient_op_typePartitionedCall-115357*'
_output_shapes
:?????????*
Tout
2?
output/StatefulPartitionedCallStatefulPartitionedCalldrop3/PartitionedCall:output:0%output_statefulpartitionedcall_args_1%output_statefulpartitionedcall_args_2*
Tout
2**
config_proto

CPU

GPU 2J 8*-
_gradient_op_typePartitionedCall-115379*
Tin
2*K
fFRD
B__inference_output_layer_call_and_return_conditional_losses_115373*'
_output_shapes
:??????????
IdentityIdentity'output/StatefulPartitionedCall:output:0^h1/StatefulPartitionedCall^h2/StatefulPartitionedCall^h3/StatefulPartitionedCall^output/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*F
_input_shapes5
3:?????????)::::::::28
h2/StatefulPartitionedCallh2/StatefulPartitionedCall28
h3/StatefulPartitionedCallh3/StatefulPartitionedCall28
h1/StatefulPartitionedCallh1/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:& "
 
_user_specified_nameinputs: : : : : : : : 
?Q
?
B__inference_DeepNN_layer_call_and_return_conditional_losses_115582

inputs%
!h1_matmul_readvariableop_resource&
"h1_biasadd_readvariableop_resource%
!h2_matmul_readvariableop_resource&
"h2_biasadd_readvariableop_resource%
!h3_matmul_readvariableop_resource&
"h3_biasadd_readvariableop_resource)
%output_matmul_readvariableop_resource*
&output_biasadd_readvariableop_resource
identity??h1/BiasAdd/ReadVariableOp?h1/MatMul/ReadVariableOp?h2/BiasAdd/ReadVariableOp?h2/MatMul/ReadVariableOp?h3/BiasAdd/ReadVariableOp?h3/MatMul/ReadVariableOp?output/BiasAdd/ReadVariableOp?output/MatMul/ReadVariableOp?
h1/MatMul/ReadVariableOpReadVariableOp!h1_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:)o
	h1/MatMulMatMulinputs h1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
h1/BiasAdd/ReadVariableOpReadVariableOp"h1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:*
dtype0

h1/BiasAddBiasAddh1/MatMul:product:0!h1/BiasAdd/ReadVariableOp:value:0*'
_output_shapes
:?????????*
T0V
h1/ReluReluh1/BiasAdd:output:0*'
_output_shapes
:?????????*
T0W
drop1/dropout/rateConst*
_output_shapes
: *
valueB
 *??L>*
dtype0X
drop1/dropout/ShapeShapeh1/Relu:activations:0*
_output_shapes
:*
T0e
 drop1/dropout/random_uniform/minConst*
_output_shapes
: *
valueB
 *    *
dtype0e
 drop1/dropout/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
*drop1/dropout/random_uniform/RandomUniformRandomUniformdrop1/dropout/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0?
 drop1/dropout/random_uniform/subSub)drop1/dropout/random_uniform/max:output:0)drop1/dropout/random_uniform/min:output:0*
T0*
_output_shapes
: ?
 drop1/dropout/random_uniform/mulMul3drop1/dropout/random_uniform/RandomUniform:output:0$drop1/dropout/random_uniform/sub:z:0*'
_output_shapes
:?????????*
T0?
drop1/dropout/random_uniformAdd$drop1/dropout/random_uniform/mul:z:0)drop1/dropout/random_uniform/min:output:0*
T0*'
_output_shapes
:?????????X
drop1/dropout/sub/xConst*
valueB
 *  ??*
_output_shapes
: *
dtype0t
drop1/dropout/subSubdrop1/dropout/sub/x:output:0drop1/dropout/rate:output:0*
T0*
_output_shapes
: \
drop1/dropout/truediv/xConst*
valueB
 *  ??*
dtype0*
_output_shapes
: z
drop1/dropout/truedivRealDiv drop1/dropout/truediv/x:output:0drop1/dropout/sub:z:0*
T0*
_output_shapes
: ?
drop1/dropout/GreaterEqualGreaterEqual drop1/dropout/random_uniform:z:0drop1/dropout/rate:output:0*'
_output_shapes
:?????????*
T0|
drop1/dropout/mulMulh1/Relu:activations:0drop1/dropout/truediv:z:0*
T0*'
_output_shapes
:?????????{
drop1/dropout/CastCastdrop1/dropout/GreaterEqual:z:0*

SrcT0
*'
_output_shapes
:?????????*

DstT0{
drop1/dropout/mul_1Muldrop1/dropout/mul:z:0drop1/dropout/Cast:y:0*'
_output_shapes
:?????????*
T0?
h2/MatMul/ReadVariableOpReadVariableOp!h2_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes

:
*
dtype0?
	h2/MatMulMatMuldrop1/dropout/mul_1:z:0 h2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
?
h2/BiasAdd/ReadVariableOpReadVariableOp"h2_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:


h2/BiasAddBiasAddh2/MatMul:product:0!h2/BiasAdd/ReadVariableOp:value:0*'
_output_shapes
:?????????
*
T0V
h2/ReluReluh2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
W
drop2/dropout/rateConst*
valueB
 *??L>*
_output_shapes
: *
dtype0X
drop2/dropout/ShapeShapeh2/Relu:activations:0*
T0*
_output_shapes
:e
 drop2/dropout/random_uniform/minConst*
valueB
 *    *
_output_shapes
: *
dtype0e
 drop2/dropout/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *  ???
*drop2/dropout/random_uniform/RandomUniformRandomUniformdrop2/dropout/Shape:output:0*'
_output_shapes
:?????????
*
dtype0*
T0?
 drop2/dropout/random_uniform/subSub)drop2/dropout/random_uniform/max:output:0)drop2/dropout/random_uniform/min:output:0*
T0*
_output_shapes
: ?
 drop2/dropout/random_uniform/mulMul3drop2/dropout/random_uniform/RandomUniform:output:0$drop2/dropout/random_uniform/sub:z:0*'
_output_shapes
:?????????
*
T0?
drop2/dropout/random_uniformAdd$drop2/dropout/random_uniform/mul:z:0)drop2/dropout/random_uniform/min:output:0*
T0*'
_output_shapes
:?????????
X
drop2/dropout/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??t
drop2/dropout/subSubdrop2/dropout/sub/x:output:0drop2/dropout/rate:output:0*
_output_shapes
: *
T0\
drop2/dropout/truediv/xConst*
dtype0*
valueB
 *  ??*
_output_shapes
: z
drop2/dropout/truedivRealDiv drop2/dropout/truediv/x:output:0drop2/dropout/sub:z:0*
_output_shapes
: *
T0?
drop2/dropout/GreaterEqualGreaterEqual drop2/dropout/random_uniform:z:0drop2/dropout/rate:output:0*
T0*'
_output_shapes
:?????????
|
drop2/dropout/mulMulh2/Relu:activations:0drop2/dropout/truediv:z:0*'
_output_shapes
:?????????
*
T0{
drop2/dropout/CastCastdrop2/dropout/GreaterEqual:z:0*

SrcT0
*

DstT0*'
_output_shapes
:?????????
{
drop2/dropout/mul_1Muldrop2/dropout/mul:z:0drop2/dropout/Cast:y:0*
T0*'
_output_shapes
:?????????
?
h3/MatMul/ReadVariableOpReadVariableOp!h3_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes

:
*
dtype0?
	h3/MatMulMatMuldrop2/dropout/mul_1:z:0 h3/MatMul/ReadVariableOp:value:0*'
_output_shapes
:?????????*
T0?
h3/BiasAdd/ReadVariableOpReadVariableOp"h3_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:

h3/BiasAddBiasAddh3/MatMul:product:0!h3/BiasAdd/ReadVariableOp:value:0*'
_output_shapes
:?????????*
T0V
h3/ReluReluh3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????W
drop3/dropout/rateConst*
valueB
 *??L>*
_output_shapes
: *
dtype0X
drop3/dropout/ShapeShapeh3/Relu:activations:0*
T0*
_output_shapes
:e
 drop3/dropout/random_uniform/minConst*
_output_shapes
: *
valueB
 *    *
dtype0e
 drop3/dropout/random_uniform/maxConst*
valueB
 *  ??*
dtype0*
_output_shapes
: ?
*drop3/dropout/random_uniform/RandomUniformRandomUniformdrop3/dropout/Shape:output:0*
T0*
dtype0*'
_output_shapes
:??????????
 drop3/dropout/random_uniform/subSub)drop3/dropout/random_uniform/max:output:0)drop3/dropout/random_uniform/min:output:0*
_output_shapes
: *
T0?
 drop3/dropout/random_uniform/mulMul3drop3/dropout/random_uniform/RandomUniform:output:0$drop3/dropout/random_uniform/sub:z:0*'
_output_shapes
:?????????*
T0?
drop3/dropout/random_uniformAdd$drop3/dropout/random_uniform/mul:z:0)drop3/dropout/random_uniform/min:output:0*
T0*'
_output_shapes
:?????????X
drop3/dropout/sub/xConst*
dtype0*
_output_shapes
: *
valueB
 *  ??t
drop3/dropout/subSubdrop3/dropout/sub/x:output:0drop3/dropout/rate:output:0*
_output_shapes
: *
T0\
drop3/dropout/truediv/xConst*
valueB
 *  ??*
_output_shapes
: *
dtype0z
drop3/dropout/truedivRealDiv drop3/dropout/truediv/x:output:0drop3/dropout/sub:z:0*
_output_shapes
: *
T0?
drop3/dropout/GreaterEqualGreaterEqual drop3/dropout/random_uniform:z:0drop3/dropout/rate:output:0*
T0*'
_output_shapes
:?????????|
drop3/dropout/mulMulh3/Relu:activations:0drop3/dropout/truediv:z:0*'
_output_shapes
:?????????*
T0{
drop3/dropout/CastCastdrop3/dropout/GreaterEqual:z:0*'
_output_shapes
:?????????*

DstT0*

SrcT0
{
drop3/dropout/mul_1Muldrop3/dropout/mul:z:0drop3/dropout/Cast:y:0*
T0*'
_output_shapes
:??????????
output/MatMul/ReadVariableOpReadVariableOp%output_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes

:*
dtype0?
output/MatMulMatMuldrop3/dropout/mul_1:z:0$output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
output/BiasAdd/ReadVariableOpReadVariableOp&output_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:*
dtype0?
output/BiasAddBiasAddoutput/MatMul:product:0%output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d
output/SigmoidSigmoidoutput/BiasAdd:output:0*'
_output_shapes
:?????????*
T0?
IdentityIdentityoutput/Sigmoid:y:0^h1/BiasAdd/ReadVariableOp^h1/MatMul/ReadVariableOp^h2/BiasAdd/ReadVariableOp^h2/MatMul/ReadVariableOp^h3/BiasAdd/ReadVariableOp^h3/MatMul/ReadVariableOp^output/BiasAdd/ReadVariableOp^output/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*F
_input_shapes5
3:?????????)::::::::2>
output/BiasAdd/ReadVariableOpoutput/BiasAdd/ReadVariableOp24
h1/MatMul/ReadVariableOph1/MatMul/ReadVariableOp24
h3/MatMul/ReadVariableOph3/MatMul/ReadVariableOp26
h3/BiasAdd/ReadVariableOph3/BiasAdd/ReadVariableOp26
h2/BiasAdd/ReadVariableOph2/BiasAdd/ReadVariableOp26
h1/BiasAdd/ReadVariableOph1/BiasAdd/ReadVariableOp2<
output/MatMul/ReadVariableOpoutput/MatMul/ReadVariableOp24
h2/MatMul/ReadVariableOph2/MatMul/ReadVariableOp: : : : : : : : :& "
 
_user_specified_nameinputs
?
_
&__inference_drop3_layer_call_fn_115797

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*J
fERC
A__inference_drop3_layer_call_and_return_conditional_losses_115338**
config_proto

CPU

GPU 2J 8*-
_gradient_op_typePartitionedCall-115349*'
_output_shapes
:?????????*
Tin
2*
Tout
2?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*&
_input_shapes
:?????????22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
?
?
#__inference_h3_layer_call_fn_115767

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tout
2*G
fBR@
>__inference_h3_layer_call_and_return_conditional_losses_115301*
Tin
2*'
_output_shapes
:?????????*-
_gradient_op_typePartitionedCall-115307**
config_proto

CPU

GPU 2J 8?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*.
_input_shapes
:?????????
::22
StatefulPartitionedCallStatefulPartitionedCall: :& "
 
_user_specified_nameinputs: 
?
`
A__inference_drop3_layer_call_and_return_conditional_losses_115787

inputs
identity?Q
dropout/rateConst*
valueB
 *??L>*
_output_shapes
: *
dtype0C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:_
dropout/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *    _
dropout/random_uniform/maxConst*
dtype0*
valueB
 *  ??*
_output_shapes
: ?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
dtype0*'
_output_shapes
:?????????*
T0?
dropout/random_uniform/subSub#dropout/random_uniform/max:output:0#dropout/random_uniform/min:output:0*
_output_shapes
: *
T0?
dropout/random_uniform/mulMul-dropout/random_uniform/RandomUniform:output:0dropout/random_uniform/sub:z:0*
T0*'
_output_shapes
:??????????
dropout/random_uniformAdddropout/random_uniform/mul:z:0#dropout/random_uniform/min:output:0*'
_output_shapes
:?????????*
T0R
dropout/sub/xConst*
dtype0*
valueB
 *  ??*
_output_shapes
: b
dropout/subSubdropout/sub/x:output:0dropout/rate:output:0*
_output_shapes
: *
T0V
dropout/truediv/xConst*
dtype0*
valueB
 *  ??*
_output_shapes
: h
dropout/truedivRealDivdropout/truediv/x:output:0dropout/sub:z:0*
_output_shapes
: *
T0?
dropout/GreaterEqualGreaterEqualdropout/random_uniform:z:0dropout/rate:output:0*'
_output_shapes
:?????????*
T0a
dropout/mulMulinputsdropout/truediv:z:0*
T0*'
_output_shapes
:?????????o
dropout/CastCastdropout/GreaterEqual:z:0*'
_output_shapes
:?????????*

SrcT0
*

DstT0i
dropout/mul_1Muldropout/mul:z:0dropout/Cast:y:0*'
_output_shapes
:?????????*
T0Y
IdentityIdentitydropout/mul_1:z:0*'
_output_shapes
:?????????*
T0"
identityIdentity:output:0*&
_input_shapes
:?????????:& "
 
_user_specified_nameinputs
?=
?
__inference__traced_save_115938
file_prefix(
$savev2_h1_kernel_read_readvariableop&
"savev2_h1_bias_read_readvariableop(
$savev2_h2_kernel_read_readvariableop&
"savev2_h2_bias_read_readvariableop(
$savev2_h3_kernel_read_readvariableop&
"savev2_h3_bias_read_readvariableop,
(savev2_output_kernel_read_readvariableop*
&savev2_output_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop/
+savev2_adam_h1_kernel_m_read_readvariableop-
)savev2_adam_h1_bias_m_read_readvariableop/
+savev2_adam_h2_kernel_m_read_readvariableop-
)savev2_adam_h2_bias_m_read_readvariableop/
+savev2_adam_h3_kernel_m_read_readvariableop-
)savev2_adam_h3_bias_m_read_readvariableop3
/savev2_adam_output_kernel_m_read_readvariableop1
-savev2_adam_output_bias_m_read_readvariableop/
+savev2_adam_h1_kernel_v_read_readvariableop-
)savev2_adam_h1_bias_v_read_readvariableop/
+savev2_adam_h2_kernel_v_read_readvariableop-
)savev2_adam_h2_bias_v_read_readvariableop/
+savev2_adam_h3_kernel_v_read_readvariableop-
)savev2_adam_h3_bias_v_read_readvariableop3
/savev2_adam_output_kernel_v_read_readvariableop1
-savev2_adam_output_bias_v_read_readvariableop
savev2_1_const

identity_1??MergeV2Checkpoints?SaveV2?SaveV2_1?
StringJoin/inputs_1Const"/device:CPU:0*
dtype0*<
value3B1 B+_temp_7f86a2b7ddee44bf98542052e4ee65de/part*
_output_shapes
: s

StringJoin
StringJoinfile_prefixStringJoin/inputs_1:output:0"/device:CPU:0*
_output_shapes
: *
NL

num_shardsConst*
value	B :*
_output_shapes
: *
dtype0f
ShardedFilename/shardConst"/device:CPU:0*
dtype0*
_output_shapes
: *
value	B : ?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*?
value?B?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
dtype0?
SaveV2/shape_and_slicesConst"/device:CPU:0*Q
valueHBFB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
_output_shapes
:*
dtype0?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0$savev2_h1_kernel_read_readvariableop"savev2_h1_bias_read_readvariableop$savev2_h2_kernel_read_readvariableop"savev2_h2_bias_read_readvariableop$savev2_h3_kernel_read_readvariableop"savev2_h3_bias_read_readvariableop(savev2_output_kernel_read_readvariableop&savev2_output_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop+savev2_adam_h1_kernel_m_read_readvariableop)savev2_adam_h1_bias_m_read_readvariableop+savev2_adam_h2_kernel_m_read_readvariableop)savev2_adam_h2_bias_m_read_readvariableop+savev2_adam_h3_kernel_m_read_readvariableop)savev2_adam_h3_bias_m_read_readvariableop/savev2_adam_output_kernel_m_read_readvariableop-savev2_adam_output_bias_m_read_readvariableop+savev2_adam_h1_kernel_v_read_readvariableop)savev2_adam_h1_bias_v_read_readvariableop+savev2_adam_h2_kernel_v_read_readvariableop)savev2_adam_h2_bias_v_read_readvariableop+savev2_adam_h3_kernel_v_read_readvariableop)savev2_adam_h3_bias_v_read_readvariableop/savev2_adam_output_kernel_v_read_readvariableop-savev2_adam_output_bias_v_read_readvariableop"/device:CPU:0*
_output_shapes
 *-
dtypes#
!2	h
ShardedFilename_1/shardConst"/device:CPU:0*
value	B :*
dtype0*
_output_shapes
: ?
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ?
SaveV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH*
dtype0q
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B ?
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
_output_shapes
 *
dtypes
2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
_output_shapes
:*
T0*
N?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
_output_shapes
: *
T0s

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0*?
_input_shapes?
?: :)::
:
:
:::: : : : : : : :)::
:
:
::::)::
:
:
:::: 2
SaveV2_1SaveV2_12(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2SaveV2:+ '
%
_user_specified_namefile_prefix: : : : : : : : :	 :
 : : : : : : : : : : : : : : : : : : : : : :  
?
_
&__inference_drop2_layer_call_fn_115744

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*-
_gradient_op_typePartitionedCall-115277*
Tout
2**
config_proto

CPU

GPU 2J 8*J
fERC
A__inference_drop2_layer_call_and_return_conditional_losses_115266*'
_output_shapes
:?????????
*
Tin
2?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*'
_output_shapes
:?????????
*
T0"
identityIdentity:output:0*&
_input_shapes
:?????????
22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
?
B
&__inference_drop3_layer_call_fn_115802

inputs
identity?
PartitionedCallPartitionedCallinputs**
config_proto

CPU

GPU 2J 8*-
_gradient_op_typePartitionedCall-115357*
Tout
2*J
fERC
A__inference_drop3_layer_call_and_return_conditional_losses_115345*
Tin
2*'
_output_shapes
:?????????`
IdentityIdentityPartitionedCall:output:0*'
_output_shapes
:?????????*
T0"
identityIdentity:output:0*&
_input_shapes
:?????????:& "
 
_user_specified_nameinputs
?
B
&__inference_drop2_layer_call_fn_115749

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tout
2*J
fERC
A__inference_drop2_layer_call_and_return_conditional_losses_115273**
config_proto

CPU

GPU 2J 8*-
_gradient_op_typePartitionedCall-115285*
Tin
2*'
_output_shapes
:?????????
`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????
"
identityIdentity:output:0*&
_input_shapes
:?????????
:& "
 
_user_specified_nameinputs
?

?
$__inference_signature_wrapper_115500
h1_input"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallh1_inputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8**
config_proto

CPU

GPU 2J 8**
f%R#
!__inference__wrapped_model_115140*
Tout
2*'
_output_shapes
:?????????*
Tin
2	*-
_gradient_op_typePartitionedCall-115489?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*'
_output_shapes
:?????????*
T0"
identityIdentity:output:0*F
_input_shapes5
3:?????????)::::::::22
StatefulPartitionedCallStatefulPartitionedCall: :( $
"
_user_specified_name
h1_input: : : : : : : 
?
?
'__inference_output_layer_call_fn_115820

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*K
fFRD
B__inference_output_layer_call_and_return_conditional_losses_115373*
Tin
2**
config_proto

CPU

GPU 2J 8*-
_gradient_op_typePartitionedCall-115379*
Tout
2*'
_output_shapes
:??????????
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*'
_output_shapes
:?????????*
T0"
identityIdentity:output:0*.
_input_shapes
:?????????::22
StatefulPartitionedCallStatefulPartitionedCall: :& "
 
_user_specified_nameinputs: 
?	
?
>__inference_h1_layer_call_and_return_conditional_losses_115157

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:)i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*'
_output_shapes
:?????????*
T0P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:??????????
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*'
_output_shapes
:?????????*
T0"
identityIdentity:output:0*.
_input_shapes
:?????????)::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
?
`
A__inference_drop2_layer_call_and_return_conditional_losses_115266

inputs
identity?Q
dropout/rateConst*
dtype0*
_output_shapes
: *
valueB
 *??L>C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:_
dropout/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    _
dropout/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*'
_output_shapes
:?????????
*
dtype0*
T0?
dropout/random_uniform/subSub#dropout/random_uniform/max:output:0#dropout/random_uniform/min:output:0*
_output_shapes
: *
T0?
dropout/random_uniform/mulMul-dropout/random_uniform/RandomUniform:output:0dropout/random_uniform/sub:z:0*
T0*'
_output_shapes
:?????????
?
dropout/random_uniformAdddropout/random_uniform/mul:z:0#dropout/random_uniform/min:output:0*
T0*'
_output_shapes
:?????????
R
dropout/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??b
dropout/subSubdropout/sub/x:output:0dropout/rate:output:0*
_output_shapes
: *
T0V
dropout/truediv/xConst*
valueB
 *  ??*
dtype0*
_output_shapes
: h
dropout/truedivRealDivdropout/truediv/x:output:0dropout/sub:z:0*
T0*
_output_shapes
: ?
dropout/GreaterEqualGreaterEqualdropout/random_uniform:z:0dropout/rate:output:0*'
_output_shapes
:?????????
*
T0a
dropout/mulMulinputsdropout/truediv:z:0*
T0*'
_output_shapes
:?????????
o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*'
_output_shapes
:?????????
*

SrcT0
i
dropout/mul_1Muldropout/mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:?????????
Y
IdentityIdentitydropout/mul_1:z:0*
T0*'
_output_shapes
:?????????
"
identityIdentity:output:0*&
_input_shapes
:?????????
:& "
 
_user_specified_nameinputs
?	
?
>__inference_h2_layer_call_and_return_conditional_losses_115707

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes

:
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*'
_output_shapes
:?????????
*
T0?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????
?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*'
_output_shapes
:?????????
*
T0"
identityIdentity:output:0*.
_input_shapes
:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
?	
?
>__inference_h2_layer_call_and_return_conditional_losses_115229

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes

:
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*'
_output_shapes
:?????????
*
T0?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
P
ReluReluBiasAdd:output:0*'
_output_shapes
:?????????
*
T0?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????
"
identityIdentity:output:0*.
_input_shapes
:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
?	
?
B__inference_output_layer_call_and_return_conditional_losses_115813

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:??????????
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*'
_output_shapes
:?????????*
T0"
identityIdentity:output:0*.
_input_shapes
:?????????::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp: : :& "
 
_user_specified_nameinputs
?

?
'__inference_DeepNN_layer_call_fn_115481
h1_input"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallh1_inputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8*
Tin
2	*'
_output_shapes
:?????????*
Tout
2*K
fFRD
B__inference_DeepNN_layer_call_and_return_conditional_losses_115469**
config_proto

CPU

GPU 2J 8*-
_gradient_op_typePartitionedCall-115470?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*F
_input_shapes5
3:?????????)::::::::22
StatefulPartitionedCallStatefulPartitionedCall: : : : : :( $
"
_user_specified_name
h1_input: : : 
?
_
A__inference_drop3_layer_call_and_return_conditional_losses_115345

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:?????????[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:?????????"!

identity_1Identity_1:output:0*&
_input_shapes
:?????????:& "
 
_user_specified_nameinputs
?
?
B__inference_DeepNN_layer_call_and_return_conditional_losses_115412
h1_input%
!h1_statefulpartitionedcall_args_1%
!h1_statefulpartitionedcall_args_2%
!h2_statefulpartitionedcall_args_1%
!h2_statefulpartitionedcall_args_2%
!h3_statefulpartitionedcall_args_1%
!h3_statefulpartitionedcall_args_2)
%output_statefulpartitionedcall_args_1)
%output_statefulpartitionedcall_args_2
identity??h1/StatefulPartitionedCall?h2/StatefulPartitionedCall?h3/StatefulPartitionedCall?output/StatefulPartitionedCall?
h1/StatefulPartitionedCallStatefulPartitionedCallh1_input!h1_statefulpartitionedcall_args_1!h1_statefulpartitionedcall_args_2*
Tin
2**
config_proto

CPU

GPU 2J 8*
Tout
2*-
_gradient_op_typePartitionedCall-115163*G
fBR@
>__inference_h1_layer_call_and_return_conditional_losses_115157*'
_output_shapes
:??????????
drop1/PartitionedCallPartitionedCall#h1/StatefulPartitionedCall:output:0**
config_proto

CPU

GPU 2J 8*
Tin
2*J
fERC
A__inference_drop1_layer_call_and_return_conditional_losses_115201*'
_output_shapes
:?????????*
Tout
2*-
_gradient_op_typePartitionedCall-115213?
h2/StatefulPartitionedCallStatefulPartitionedCalldrop1/PartitionedCall:output:0!h2_statefulpartitionedcall_args_1!h2_statefulpartitionedcall_args_2*
Tout
2*
Tin
2**
config_proto

CPU

GPU 2J 8*G
fBR@
>__inference_h2_layer_call_and_return_conditional_losses_115229*-
_gradient_op_typePartitionedCall-115235*'
_output_shapes
:?????????
?
drop2/PartitionedCallPartitionedCall#h2/StatefulPartitionedCall:output:0*
Tout
2*
Tin
2**
config_proto

CPU

GPU 2J 8*-
_gradient_op_typePartitionedCall-115285*'
_output_shapes
:?????????
*J
fERC
A__inference_drop2_layer_call_and_return_conditional_losses_115273?
h3/StatefulPartitionedCallStatefulPartitionedCalldrop2/PartitionedCall:output:0!h3_statefulpartitionedcall_args_1!h3_statefulpartitionedcall_args_2*
Tin
2*'
_output_shapes
:?????????*-
_gradient_op_typePartitionedCall-115307*G
fBR@
>__inference_h3_layer_call_and_return_conditional_losses_115301*
Tout
2**
config_proto

CPU

GPU 2J 8?
drop3/PartitionedCallPartitionedCall#h3/StatefulPartitionedCall:output:0*-
_gradient_op_typePartitionedCall-115357*
Tin
2*J
fERC
A__inference_drop3_layer_call_and_return_conditional_losses_115345*'
_output_shapes
:?????????*
Tout
2**
config_proto

CPU

GPU 2J 8?
output/StatefulPartitionedCallStatefulPartitionedCalldrop3/PartitionedCall:output:0%output_statefulpartitionedcall_args_1%output_statefulpartitionedcall_args_2*K
fFRD
B__inference_output_layer_call_and_return_conditional_losses_115373*'
_output_shapes
:?????????**
config_proto

CPU

GPU 2J 8*
Tout
2*-
_gradient_op_typePartitionedCall-115379*
Tin
2?
IdentityIdentity'output/StatefulPartitionedCall:output:0^h1/StatefulPartitionedCall^h2/StatefulPartitionedCall^h3/StatefulPartitionedCall^output/StatefulPartitionedCall*'
_output_shapes
:?????????*
T0"
identityIdentity:output:0*F
_input_shapes5
3:?????????)::::::::2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall28
h1/StatefulPartitionedCallh1/StatefulPartitionedCall28
h2/StatefulPartitionedCallh2/StatefulPartitionedCall28
h3/StatefulPartitionedCallh3/StatefulPartitionedCall:( $
"
_user_specified_name
h1_input: : : : : : : : 
?)
?
!__inference__wrapped_model_115140
h1_input,
(deepnn_h1_matmul_readvariableop_resource-
)deepnn_h1_biasadd_readvariableop_resource,
(deepnn_h2_matmul_readvariableop_resource-
)deepnn_h2_biasadd_readvariableop_resource,
(deepnn_h3_matmul_readvariableop_resource-
)deepnn_h3_biasadd_readvariableop_resource0
,deepnn_output_matmul_readvariableop_resource1
-deepnn_output_biasadd_readvariableop_resource
identity?? DeepNN/h1/BiasAdd/ReadVariableOp?DeepNN/h1/MatMul/ReadVariableOp? DeepNN/h2/BiasAdd/ReadVariableOp?DeepNN/h2/MatMul/ReadVariableOp? DeepNN/h3/BiasAdd/ReadVariableOp?DeepNN/h3/MatMul/ReadVariableOp?$DeepNN/output/BiasAdd/ReadVariableOp?#DeepNN/output/MatMul/ReadVariableOp?
DeepNN/h1/MatMul/ReadVariableOpReadVariableOp(deepnn_h1_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:)
DeepNN/h1/MatMulMatMulh1_input'DeepNN/h1/MatMul/ReadVariableOp:value:0*'
_output_shapes
:?????????*
T0?
 DeepNN/h1/BiasAdd/ReadVariableOpReadVariableOp)deepnn_h1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:?
DeepNN/h1/BiasAddBiasAddDeepNN/h1/MatMul:product:0(DeepNN/h1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d
DeepNN/h1/ReluReluDeepNN/h1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????q
DeepNN/drop1/IdentityIdentityDeepNN/h1/Relu:activations:0*'
_output_shapes
:?????????*
T0?
DeepNN/h2/MatMul/ReadVariableOpReadVariableOp(deepnn_h2_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:
?
DeepNN/h2/MatMulMatMulDeepNN/drop1/Identity:output:0'DeepNN/h2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
?
 DeepNN/h2/BiasAdd/ReadVariableOpReadVariableOp)deepnn_h2_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:
?
DeepNN/h2/BiasAddBiasAddDeepNN/h2/MatMul:product:0(DeepNN/h2/BiasAdd/ReadVariableOp:value:0*'
_output_shapes
:?????????
*
T0d
DeepNN/h2/ReluReluDeepNN/h2/BiasAdd:output:0*'
_output_shapes
:?????????
*
T0q
DeepNN/drop2/IdentityIdentityDeepNN/h2/Relu:activations:0*'
_output_shapes
:?????????
*
T0?
DeepNN/h3/MatMul/ReadVariableOpReadVariableOp(deepnn_h3_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes

:
*
dtype0?
DeepNN/h3/MatMulMatMulDeepNN/drop2/Identity:output:0'DeepNN/h3/MatMul/ReadVariableOp:value:0*'
_output_shapes
:?????????*
T0?
 DeepNN/h3/BiasAdd/ReadVariableOpReadVariableOp)deepnn_h3_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:?
DeepNN/h3/BiasAddBiasAddDeepNN/h3/MatMul:product:0(DeepNN/h3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d
DeepNN/h3/ReluReluDeepNN/h3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????q
DeepNN/drop3/IdentityIdentityDeepNN/h3/Relu:activations:0*'
_output_shapes
:?????????*
T0?
#DeepNN/output/MatMul/ReadVariableOpReadVariableOp,deepnn_output_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes

:*
dtype0?
DeepNN/output/MatMulMatMulDeepNN/drop3/Identity:output:0+DeepNN/output/MatMul/ReadVariableOp:value:0*'
_output_shapes
:?????????*
T0?
$DeepNN/output/BiasAdd/ReadVariableOpReadVariableOp-deepnn_output_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:*
dtype0?
DeepNN/output/BiasAddBiasAddDeepNN/output/MatMul:product:0,DeepNN/output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
DeepNN/output/SigmoidSigmoidDeepNN/output/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
IdentityIdentityDeepNN/output/Sigmoid:y:0!^DeepNN/h1/BiasAdd/ReadVariableOp ^DeepNN/h1/MatMul/ReadVariableOp!^DeepNN/h2/BiasAdd/ReadVariableOp ^DeepNN/h2/MatMul/ReadVariableOp!^DeepNN/h3/BiasAdd/ReadVariableOp ^DeepNN/h3/MatMul/ReadVariableOp%^DeepNN/output/BiasAdd/ReadVariableOp$^DeepNN/output/MatMul/ReadVariableOp*'
_output_shapes
:?????????*
T0"
identityIdentity:output:0*F
_input_shapes5
3:?????????)::::::::2L
$DeepNN/output/BiasAdd/ReadVariableOp$DeepNN/output/BiasAdd/ReadVariableOp2D
 DeepNN/h3/BiasAdd/ReadVariableOp DeepNN/h3/BiasAdd/ReadVariableOp2J
#DeepNN/output/MatMul/ReadVariableOp#DeepNN/output/MatMul/ReadVariableOp2B
DeepNN/h2/MatMul/ReadVariableOpDeepNN/h2/MatMul/ReadVariableOp2D
 DeepNN/h2/BiasAdd/ReadVariableOp DeepNN/h2/BiasAdd/ReadVariableOp2D
 DeepNN/h1/BiasAdd/ReadVariableOp DeepNN/h1/BiasAdd/ReadVariableOp2B
DeepNN/h1/MatMul/ReadVariableOpDeepNN/h1/MatMul/ReadVariableOp2B
DeepNN/h3/MatMul/ReadVariableOpDeepNN/h3/MatMul/ReadVariableOp: : : : : :( $
"
_user_specified_name
h1_input: : : 
?
_
A__inference_drop1_layer_call_and_return_conditional_losses_115686

inputs

identity_1N
IdentityIdentityinputs*'
_output_shapes
:?????????*
T0[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:?????????"!

identity_1Identity_1:output:0*&
_input_shapes
:?????????:& "
 
_user_specified_nameinputs
?	
?
B__inference_output_layer_call_and_return_conditional_losses_115373

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*'
_output_shapes
:?????????*
T0?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*'
_output_shapes
:?????????*
T0V
SigmoidSigmoidBiasAdd:output:0*'
_output_shapes
:?????????*
T0?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*.
_input_shapes
:?????????::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
?	
?
>__inference_h3_layer_call_and_return_conditional_losses_115760

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes

:
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:??????????
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*'
_output_shapes
:?????????*
T0"
identityIdentity:output:0*.
_input_shapes
:?????????
::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
?
`
A__inference_drop1_layer_call_and_return_conditional_losses_115194

inputs
identity?Q
dropout/rateConst*
valueB
 *??L>*
dtype0*
_output_shapes
: C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:_
dropout/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *    _
dropout/random_uniform/maxConst*
valueB
 *  ??*
_output_shapes
: *
dtype0?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0?
dropout/random_uniform/subSub#dropout/random_uniform/max:output:0#dropout/random_uniform/min:output:0*
T0*
_output_shapes
: ?
dropout/random_uniform/mulMul-dropout/random_uniform/RandomUniform:output:0dropout/random_uniform/sub:z:0*'
_output_shapes
:?????????*
T0?
dropout/random_uniformAdddropout/random_uniform/mul:z:0#dropout/random_uniform/min:output:0*'
_output_shapes
:?????????*
T0R
dropout/sub/xConst*
valueB
 *  ??*
dtype0*
_output_shapes
: b
dropout/subSubdropout/sub/x:output:0dropout/rate:output:0*
T0*
_output_shapes
: V
dropout/truediv/xConst*
valueB
 *  ??*
_output_shapes
: *
dtype0h
dropout/truedivRealDivdropout/truediv/x:output:0dropout/sub:z:0*
_output_shapes
: *
T0?
dropout/GreaterEqualGreaterEqualdropout/random_uniform:z:0dropout/rate:output:0*'
_output_shapes
:?????????*
T0a
dropout/mulMulinputsdropout/truediv:z:0*
T0*'
_output_shapes
:?????????o
dropout/CastCastdropout/GreaterEqual:z:0*'
_output_shapes
:?????????*

SrcT0
*

DstT0i
dropout/mul_1Muldropout/mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:?????????Y
IdentityIdentitydropout/mul_1:z:0*'
_output_shapes
:?????????*
T0"
identityIdentity:output:0*&
_input_shapes
:?????????:& "
 
_user_specified_nameinputs
?$
?
B__inference_DeepNN_layer_call_and_return_conditional_losses_115617

inputs%
!h1_matmul_readvariableop_resource&
"h1_biasadd_readvariableop_resource%
!h2_matmul_readvariableop_resource&
"h2_biasadd_readvariableop_resource%
!h3_matmul_readvariableop_resource&
"h3_biasadd_readvariableop_resource)
%output_matmul_readvariableop_resource*
&output_biasadd_readvariableop_resource
identity??h1/BiasAdd/ReadVariableOp?h1/MatMul/ReadVariableOp?h2/BiasAdd/ReadVariableOp?h2/MatMul/ReadVariableOp?h3/BiasAdd/ReadVariableOp?h3/MatMul/ReadVariableOp?output/BiasAdd/ReadVariableOp?output/MatMul/ReadVariableOp?
h1/MatMul/ReadVariableOpReadVariableOp!h1_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes

:)*
dtype0o
	h1/MatMulMatMulinputs h1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
h1/BiasAdd/ReadVariableOpReadVariableOp"h1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:*
dtype0

h1/BiasAddBiasAddh1/MatMul:product:0!h1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????V
h1/ReluReluh1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????c
drop1/IdentityIdentityh1/Relu:activations:0*'
_output_shapes
:?????????*
T0?
h2/MatMul/ReadVariableOpReadVariableOp!h2_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes

:
*
dtype0?
	h2/MatMulMatMuldrop1/Identity:output:0 h2/MatMul/ReadVariableOp:value:0*'
_output_shapes
:?????????
*
T0?
h2/BiasAdd/ReadVariableOpReadVariableOp"h2_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:
*
dtype0

h2/BiasAddBiasAddh2/MatMul:product:0!h2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
V
h2/ReluReluh2/BiasAdd:output:0*'
_output_shapes
:?????????
*
T0c
drop2/IdentityIdentityh2/Relu:activations:0*
T0*'
_output_shapes
:?????????
?
h3/MatMul/ReadVariableOpReadVariableOp!h3_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:
?
	h3/MatMulMatMuldrop2/Identity:output:0 h3/MatMul/ReadVariableOp:value:0*'
_output_shapes
:?????????*
T0?
h3/BiasAdd/ReadVariableOpReadVariableOp"h3_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:*
dtype0

h3/BiasAddBiasAddh3/MatMul:product:0!h3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????V
h3/ReluReluh3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????c
drop3/IdentityIdentityh3/Relu:activations:0*'
_output_shapes
:?????????*
T0?
output/MatMul/ReadVariableOpReadVariableOp%output_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes

:*
dtype0?
output/MatMulMatMuldrop3/Identity:output:0$output/MatMul/ReadVariableOp:value:0*'
_output_shapes
:?????????*
T0?
output/BiasAdd/ReadVariableOpReadVariableOp&output_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:?
output/BiasAddBiasAddoutput/MatMul:product:0%output/BiasAdd/ReadVariableOp:value:0*'
_output_shapes
:?????????*
T0d
output/SigmoidSigmoidoutput/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
IdentityIdentityoutput/Sigmoid:y:0^h1/BiasAdd/ReadVariableOp^h1/MatMul/ReadVariableOp^h2/BiasAdd/ReadVariableOp^h2/MatMul/ReadVariableOp^h3/BiasAdd/ReadVariableOp^h3/MatMul/ReadVariableOp^output/BiasAdd/ReadVariableOp^output/MatMul/ReadVariableOp*'
_output_shapes
:?????????*
T0"
identityIdentity:output:0*F
_input_shapes5
3:?????????)::::::::26
h1/BiasAdd/ReadVariableOph1/BiasAdd/ReadVariableOp2<
output/MatMul/ReadVariableOpoutput/MatMul/ReadVariableOp24
h2/MatMul/ReadVariableOph2/MatMul/ReadVariableOp24
h1/MatMul/ReadVariableOph1/MatMul/ReadVariableOp2>
output/BiasAdd/ReadVariableOpoutput/BiasAdd/ReadVariableOp24
h3/MatMul/ReadVariableOph3/MatMul/ReadVariableOp26
h3/BiasAdd/ReadVariableOph3/BiasAdd/ReadVariableOp26
h2/BiasAdd/ReadVariableOph2/BiasAdd/ReadVariableOp:& "
 
_user_specified_nameinputs: : : : : : : : 
?
_
A__inference_drop2_layer_call_and_return_conditional_losses_115273

inputs

identity_1N
IdentityIdentityinputs*'
_output_shapes
:?????????
*
T0[

Identity_1IdentityIdentity:output:0*'
_output_shapes
:?????????
*
T0"!

identity_1Identity_1:output:0*&
_input_shapes
:?????????
:& "
 
_user_specified_nameinputs
?
B
&__inference_drop1_layer_call_fn_115696

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*J
fERC
A__inference_drop1_layer_call_and_return_conditional_losses_115201*'
_output_shapes
:?????????**
config_proto

CPU

GPU 2J 8*-
_gradient_op_typePartitionedCall-115213*
Tout
2`
IdentityIdentityPartitionedCall:output:0*'
_output_shapes
:?????????*
T0"
identityIdentity:output:0*&
_input_shapes
:?????????:& "
 
_user_specified_nameinputs
?

?
'__inference_DeepNN_layer_call_fn_115446
h1_input"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallh1_inputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8*'
_output_shapes
:?????????*K
fFRD
B__inference_DeepNN_layer_call_and_return_conditional_losses_115434*
Tin
2	*-
_gradient_op_typePartitionedCall-115435**
config_proto

CPU

GPU 2J 8*
Tout
2?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*'
_output_shapes
:?????????*
T0"
identityIdentity:output:0*F
_input_shapes5
3:?????????)::::::::22
StatefulPartitionedCallStatefulPartitionedCall: : : : : : : : :( $
"
_user_specified_name
h1_input
?
_
A__inference_drop3_layer_call_and_return_conditional_losses_115792

inputs

identity_1N
IdentityIdentityinputs*'
_output_shapes
:?????????*
T0[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:?????????"!

identity_1Identity_1:output:0*&
_input_shapes
:?????????:& "
 
_user_specified_nameinputs
?!
?
B__inference_DeepNN_layer_call_and_return_conditional_losses_115391
h1_input%
!h1_statefulpartitionedcall_args_1%
!h1_statefulpartitionedcall_args_2%
!h2_statefulpartitionedcall_args_1%
!h2_statefulpartitionedcall_args_2%
!h3_statefulpartitionedcall_args_1%
!h3_statefulpartitionedcall_args_2)
%output_statefulpartitionedcall_args_1)
%output_statefulpartitionedcall_args_2
identity??drop1/StatefulPartitionedCall?drop2/StatefulPartitionedCall?drop3/StatefulPartitionedCall?h1/StatefulPartitionedCall?h2/StatefulPartitionedCall?h3/StatefulPartitionedCall?output/StatefulPartitionedCall?
h1/StatefulPartitionedCallStatefulPartitionedCallh1_input!h1_statefulpartitionedcall_args_1!h1_statefulpartitionedcall_args_2*
Tin
2*
Tout
2**
config_proto

CPU

GPU 2J 8*G
fBR@
>__inference_h1_layer_call_and_return_conditional_losses_115157*-
_gradient_op_typePartitionedCall-115163*'
_output_shapes
:??????????
drop1/StatefulPartitionedCallStatefulPartitionedCall#h1/StatefulPartitionedCall:output:0*J
fERC
A__inference_drop1_layer_call_and_return_conditional_losses_115194*
Tout
2*'
_output_shapes
:?????????*-
_gradient_op_typePartitionedCall-115205**
config_proto

CPU

GPU 2J 8*
Tin
2?
h2/StatefulPartitionedCallStatefulPartitionedCall&drop1/StatefulPartitionedCall:output:0!h2_statefulpartitionedcall_args_1!h2_statefulpartitionedcall_args_2*-
_gradient_op_typePartitionedCall-115235*
Tout
2**
config_proto

CPU

GPU 2J 8*'
_output_shapes
:?????????
*G
fBR@
>__inference_h2_layer_call_and_return_conditional_losses_115229*
Tin
2?
drop2/StatefulPartitionedCallStatefulPartitionedCall#h2/StatefulPartitionedCall:output:0^drop1/StatefulPartitionedCall*J
fERC
A__inference_drop2_layer_call_and_return_conditional_losses_115266*
Tout
2*'
_output_shapes
:?????????
**
config_proto

CPU

GPU 2J 8*-
_gradient_op_typePartitionedCall-115277*
Tin
2?
h3/StatefulPartitionedCallStatefulPartitionedCall&drop2/StatefulPartitionedCall:output:0!h3_statefulpartitionedcall_args_1!h3_statefulpartitionedcall_args_2*
Tin
2*'
_output_shapes
:?????????*G
fBR@
>__inference_h3_layer_call_and_return_conditional_losses_115301*
Tout
2*-
_gradient_op_typePartitionedCall-115307**
config_proto

CPU

GPU 2J 8?
drop3/StatefulPartitionedCallStatefulPartitionedCall#h3/StatefulPartitionedCall:output:0^drop2/StatefulPartitionedCall**
config_proto

CPU

GPU 2J 8*-
_gradient_op_typePartitionedCall-115349*
Tin
2*'
_output_shapes
:?????????*J
fERC
A__inference_drop3_layer_call_and_return_conditional_losses_115338*
Tout
2?
output/StatefulPartitionedCallStatefulPartitionedCall&drop3/StatefulPartitionedCall:output:0%output_statefulpartitionedcall_args_1%output_statefulpartitionedcall_args_2*
Tout
2*
Tin
2**
config_proto

CPU

GPU 2J 8*-
_gradient_op_typePartitionedCall-115379*'
_output_shapes
:?????????*K
fFRD
B__inference_output_layer_call_and_return_conditional_losses_115373?
IdentityIdentity'output/StatefulPartitionedCall:output:0^drop1/StatefulPartitionedCall^drop2/StatefulPartitionedCall^drop3/StatefulPartitionedCall^h1/StatefulPartitionedCall^h2/StatefulPartitionedCall^h3/StatefulPartitionedCall^output/StatefulPartitionedCall*'
_output_shapes
:?????????*
T0"
identityIdentity:output:0*F
_input_shapes5
3:?????????)::::::::2>
drop3/StatefulPartitionedCalldrop3/StatefulPartitionedCall28
h1/StatefulPartitionedCallh1/StatefulPartitionedCall28
h2/StatefulPartitionedCallh2/StatefulPartitionedCall28
h3/StatefulPartitionedCallh3/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall2>
drop1/StatefulPartitionedCalldrop1/StatefulPartitionedCall2>
drop2/StatefulPartitionedCalldrop2/StatefulPartitionedCall: : :( $
"
_user_specified_name
h1_input: : : : : : 
?
`
A__inference_drop1_layer_call_and_return_conditional_losses_115681

inputs
identity?Q
dropout/rateConst*
dtype0*
valueB
 *??L>*
_output_shapes
: C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:_
dropout/random_uniform/minConst*
_output_shapes
: *
valueB
 *    *
dtype0_
dropout/random_uniform/maxConst*
_output_shapes
: *
valueB
 *  ??*
dtype0?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
dtype0*'
_output_shapes
:?????????*
T0?
dropout/random_uniform/subSub#dropout/random_uniform/max:output:0#dropout/random_uniform/min:output:0*
_output_shapes
: *
T0?
dropout/random_uniform/mulMul-dropout/random_uniform/RandomUniform:output:0dropout/random_uniform/sub:z:0*
T0*'
_output_shapes
:??????????
dropout/random_uniformAdddropout/random_uniform/mul:z:0#dropout/random_uniform/min:output:0*
T0*'
_output_shapes
:?????????R
dropout/sub/xConst*
_output_shapes
: *
valueB
 *  ??*
dtype0b
dropout/subSubdropout/sub/x:output:0dropout/rate:output:0*
_output_shapes
: *
T0V
dropout/truediv/xConst*
dtype0*
valueB
 *  ??*
_output_shapes
: h
dropout/truedivRealDivdropout/truediv/x:output:0dropout/sub:z:0*
T0*
_output_shapes
: ?
dropout/GreaterEqualGreaterEqualdropout/random_uniform:z:0dropout/rate:output:0*
T0*'
_output_shapes
:?????????a
dropout/mulMulinputsdropout/truediv:z:0*'
_output_shapes
:?????????*
T0o
dropout/CastCastdropout/GreaterEqual:z:0*

SrcT0
*'
_output_shapes
:?????????*

DstT0i
dropout/mul_1Muldropout/mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:?????????Y
IdentityIdentitydropout/mul_1:z:0*'
_output_shapes
:?????????*
T0"
identityIdentity:output:0*&
_input_shapes
:?????????:& "
 
_user_specified_nameinputs
?
`
A__inference_drop3_layer_call_and_return_conditional_losses_115338

inputs
identity?Q
dropout/rateConst*
valueB
 *??L>*
_output_shapes
: *
dtype0C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:_
dropout/random_uniform/minConst*
valueB
 *    *
_output_shapes
: *
dtype0_
dropout/random_uniform/maxConst*
_output_shapes
: *
valueB
 *  ??*
dtype0?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*'
_output_shapes
:?????????*
dtype0*
T0?
dropout/random_uniform/subSub#dropout/random_uniform/max:output:0#dropout/random_uniform/min:output:0*
_output_shapes
: *
T0?
dropout/random_uniform/mulMul-dropout/random_uniform/RandomUniform:output:0dropout/random_uniform/sub:z:0*
T0*'
_output_shapes
:??????????
dropout/random_uniformAdddropout/random_uniform/mul:z:0#dropout/random_uniform/min:output:0*'
_output_shapes
:?????????*
T0R
dropout/sub/xConst*
_output_shapes
: *
valueB
 *  ??*
dtype0b
dropout/subSubdropout/sub/x:output:0dropout/rate:output:0*
T0*
_output_shapes
: V
dropout/truediv/xConst*
valueB
 *  ??*
dtype0*
_output_shapes
: h
dropout/truedivRealDivdropout/truediv/x:output:0dropout/sub:z:0*
T0*
_output_shapes
: ?
dropout/GreaterEqualGreaterEqualdropout/random_uniform:z:0dropout/rate:output:0*
T0*'
_output_shapes
:?????????a
dropout/mulMulinputsdropout/truediv:z:0*'
_output_shapes
:?????????*
T0o
dropout/CastCastdropout/GreaterEqual:z:0*

SrcT0
*

DstT0*'
_output_shapes
:?????????i
dropout/mul_1Muldropout/mul:z:0dropout/Cast:y:0*'
_output_shapes
:?????????*
T0Y
IdentityIdentitydropout/mul_1:z:0*'
_output_shapes
:?????????*
T0"
identityIdentity:output:0*&
_input_shapes
:?????????:& "
 
_user_specified_nameinputs
?
_
&__inference_drop1_layer_call_fn_115691

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*'
_output_shapes
:?????????*J
fERC
A__inference_drop1_layer_call_and_return_conditional_losses_115194*
Tin
2*
Tout
2*-
_gradient_op_typePartitionedCall-115205**
config_proto

CPU

GPU 2J 8?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*&
_input_shapes
:?????????22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
?
_
A__inference_drop2_layer_call_and_return_conditional_losses_115739

inputs

identity_1N
IdentityIdentityinputs*'
_output_shapes
:?????????
*
T0[

Identity_1IdentityIdentity:output:0*'
_output_shapes
:?????????
*
T0"!

identity_1Identity_1:output:0*&
_input_shapes
:?????????
:& "
 
_user_specified_nameinputs"wL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
=
h1_input1
serving_default_h1_input:0?????????):
output0
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
?+
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer_with_weights-2
layer-5
layer-6
layer_with_weights-3
layer-7
		optimizer

trainable_variables
	variables
regularization_losses
	keras_api

signatures
|_default_save_signature
}__call__
*~&call_and_return_all_conditional_losses"?(
_tf_keras_sequential?({"class_name": "Sequential", "name": "DeepNN", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "DeepNN", "layers": [{"class_name": "Dense", "config": {"name": "h1", "trainable": true, "batch_input_shape": [null, 41], "dtype": "float32", "units": 21, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "drop1", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "h2", "trainable": true, "dtype": "float32", "units": 10, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "drop2", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "h3", "trainable": true, "dtype": "float32", "units": 5, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "drop3", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "output", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 41}}}, "keras_version": "2.2.4-tf", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "DeepNN", "layers": [{"class_name": "Dense", "config": {"name": "h1", "trainable": true, "batch_input_shape": [null, 41], "dtype": "float32", "units": 21, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "drop1", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "h2", "trainable": true, "dtype": "float32", "units": 10, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "drop2", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "h3", "trainable": true, "dtype": "float32", "units": 5, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "drop3", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "output", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, "training_config": {"loss": "mean_absolute_error", "metrics": ["R2"], "weighted_metrics": null, "sample_weight_mode": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
?
trainable_variables
	variables
regularization_losses
	keras_api
__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "InputLayer", "name": "h1_input", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": [null, 41], "config": {"batch_input_shape": [null, 41], "dtype": "float32", "sparse": false, "name": "h1_input"}}
?

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "h1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": [null, 41], "config": {"name": "h1", "trainable": true, "batch_input_shape": [null, 41], "dtype": "float32", "units": 21, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 41}}}}
?
trainable_variables
	variables
regularization_losses
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dropout", "name": "drop1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "drop1", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}
?

kernel
bias
trainable_variables
 	variables
!regularization_losses
"	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "h2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "h2", "trainable": true, "dtype": "float32", "units": 10, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 21}}}}
?
#trainable_variables
$	variables
%regularization_losses
&	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dropout", "name": "drop2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "drop2", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}
?

'kernel
(bias
)trainable_variables
*	variables
+regularization_losses
,	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "h3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "h3", "trainable": true, "dtype": "float32", "units": 5, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 10}}}}
?
-trainable_variables
.	variables
/regularization_losses
0	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dropout", "name": "drop3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "drop3", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}
?

1kernel
2bias
3trainable_variables
4	variables
5regularization_losses
6	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "output", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "output", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 5}}}}
?
7iter

8beta_1

9beta_2
	:decay
;learning_ratemlmmmnmo'mp(mq1mr2msvtvuvvvw'vx(vy1vz2v{"
	optimizer
X
0
1
2
3
'4
(5
16
27"
trackable_list_wrapper
X
0
1
2
3
'4
(5
16
27"
trackable_list_wrapper
 "
trackable_list_wrapper
?
<metrics

trainable_variables
	variables
=layer_regularization_losses
regularization_losses
>non_trainable_variables

?layers
}__call__
|_default_save_signature
*~&call_and_return_all_conditional_losses
&~"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
@metrics
trainable_variables
	variables
Alayer_regularization_losses
regularization_losses
Bnon_trainable_variables

Clayers
__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:)2	h1/kernel
:2h1/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Dmetrics
trainable_variables
	variables
Elayer_regularization_losses
regularization_losses
Fnon_trainable_variables

Glayers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Hmetrics
trainable_variables
	variables
Ilayer_regularization_losses
regularization_losses
Jnon_trainable_variables

Klayers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:
2	h2/kernel
:
2h2/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Lmetrics
trainable_variables
 	variables
Mlayer_regularization_losses
!regularization_losses
Nnon_trainable_variables

Olayers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Pmetrics
#trainable_variables
$	variables
Qlayer_regularization_losses
%regularization_losses
Rnon_trainable_variables

Slayers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:
2	h3/kernel
:2h3/bias
.
'0
(1"
trackable_list_wrapper
.
'0
(1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Tmetrics
)trainable_variables
*	variables
Ulayer_regularization_losses
+regularization_losses
Vnon_trainable_variables

Wlayers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Xmetrics
-trainable_variables
.	variables
Ylayer_regularization_losses
/regularization_losses
Znon_trainable_variables

[layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:2output/kernel
:2output/bias
.
10
21"
trackable_list_wrapper
.
10
21"
trackable_list_wrapper
 "
trackable_list_wrapper
?
\metrics
3trainable_variables
4	variables
]layer_regularization_losses
5regularization_losses
^non_trainable_variables

_layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
'
`0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Q
0
1
2
3
4
5
6"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
	atotal
	bcount
c
_fn_kwargs
dtrainable_variables
e	variables
fregularization_losses
g	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "MeanMetricWrapper", "name": "R2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "R2", "dtype": "float32"}}
:  (2total
:  (2count
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
a0
b1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
hmetrics
dtrainable_variables
e	variables
ilayer_regularization_losses
fregularization_losses
jnon_trainable_variables

klayers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
a0
b1"
trackable_list_wrapper
 "
trackable_list_wrapper
 :)2Adam/h1/kernel/m
:2Adam/h1/bias/m
 :
2Adam/h2/kernel/m
:
2Adam/h2/bias/m
 :
2Adam/h3/kernel/m
:2Adam/h3/bias/m
$:"2Adam/output/kernel/m
:2Adam/output/bias/m
 :)2Adam/h1/kernel/v
:2Adam/h1/bias/v
 :
2Adam/h2/kernel/v
:
2Adam/h2/bias/v
 :
2Adam/h3/kernel/v
:2Adam/h3/bias/v
$:"2Adam/output/kernel/v
:2Adam/output/bias/v
?2?
!__inference__wrapped_model_115140?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *'?$
"?
h1_input?????????)
?2?
'__inference_DeepNN_layer_call_fn_115446
'__inference_DeepNN_layer_call_fn_115643
'__inference_DeepNN_layer_call_fn_115630
'__inference_DeepNN_layer_call_fn_115481?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
B__inference_DeepNN_layer_call_and_return_conditional_losses_115617
B__inference_DeepNN_layer_call_and_return_conditional_losses_115391
B__inference_DeepNN_layer_call_and_return_conditional_losses_115582
B__inference_DeepNN_layer_call_and_return_conditional_losses_115412?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkwjkwargs
defaults? 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkwjkwargs
defaults? 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?2?
#__inference_h1_layer_call_fn_115661?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
>__inference_h1_layer_call_and_return_conditional_losses_115654?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
&__inference_drop1_layer_call_fn_115696
&__inference_drop1_layer_call_fn_115691?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
A__inference_drop1_layer_call_and_return_conditional_losses_115681
A__inference_drop1_layer_call_and_return_conditional_losses_115686?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
#__inference_h2_layer_call_fn_115714?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
>__inference_h2_layer_call_and_return_conditional_losses_115707?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
&__inference_drop2_layer_call_fn_115744
&__inference_drop2_layer_call_fn_115749?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
A__inference_drop2_layer_call_and_return_conditional_losses_115734
A__inference_drop2_layer_call_and_return_conditional_losses_115739?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
#__inference_h3_layer_call_fn_115767?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
>__inference_h3_layer_call_and_return_conditional_losses_115760?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
&__inference_drop3_layer_call_fn_115797
&__inference_drop3_layer_call_fn_115802?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
A__inference_drop3_layer_call_and_return_conditional_losses_115787
A__inference_drop3_layer_call_and_return_conditional_losses_115792?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
'__inference_output_layer_call_fn_115820?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_output_layer_call_and_return_conditional_losses_115813?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
4B2
$__inference_signature_wrapper_115500h1_input
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkwjkwargs
defaults? 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkwjkwargs
defaults? 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 ?
>__inference_h2_layer_call_and_return_conditional_losses_115707\/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????

? ?
A__inference_drop2_layer_call_and_return_conditional_losses_115734\3?0
)?&
 ?
inputs?????????

p
? "%?"
?
0?????????

? ?
A__inference_drop3_layer_call_and_return_conditional_losses_115792\3?0
)?&
 ?
inputs?????????
p 
? "%?"
?
0?????????
? v
#__inference_h2_layer_call_fn_115714O/?,
%?"
 ?
inputs?????????
? "??????????
?
B__inference_DeepNN_layer_call_and_return_conditional_losses_115391l'(129?6
/?,
"?
h1_input?????????)
p

 
? "%?"
?
0?????????
? ?
A__inference_drop3_layer_call_and_return_conditional_losses_115787\3?0
)?&
 ?
inputs?????????
p
? "%?"
?
0?????????
? ?
B__inference_DeepNN_layer_call_and_return_conditional_losses_115582j'(127?4
-?*
 ?
inputs?????????)
p

 
? "%?"
?
0?????????
? ?
B__inference_output_layer_call_and_return_conditional_losses_115813\12/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? ?
!__inference__wrapped_model_115140n'(121?.
'?$
"?
h1_input?????????)
? "/?,
*
output ?
output??????????
$__inference_signature_wrapper_115500z'(12=?:
? 
3?0
.
h1_input"?
h1_input?????????)"/?,
*
output ?
output?????????z
'__inference_output_layer_call_fn_115820O12/?,
%?"
 ?
inputs?????????
? "??????????y
&__inference_drop3_layer_call_fn_115797O3?0
)?&
 ?
inputs?????????
p
? "??????????y
&__inference_drop1_layer_call_fn_115691O3?0
)?&
 ?
inputs?????????
p
? "???????????
A__inference_drop1_layer_call_and_return_conditional_losses_115681\3?0
)?&
 ?
inputs?????????
p
? "%?"
?
0?????????
? ?
>__inference_h3_layer_call_and_return_conditional_losses_115760\'(/?,
%?"
 ?
inputs?????????

? "%?"
?
0?????????
? y
&__inference_drop2_layer_call_fn_115749O3?0
)?&
 ?
inputs?????????

p 
? "??????????
y
&__inference_drop1_layer_call_fn_115696O3?0
)?&
 ?
inputs?????????
p 
? "???????????
A__inference_drop1_layer_call_and_return_conditional_losses_115686\3?0
)?&
 ?
inputs?????????
p 
? "%?"
?
0?????????
? ?
'__inference_DeepNN_layer_call_fn_115630]'(127?4
-?*
 ?
inputs?????????)
p

 
? "??????????y
&__inference_drop2_layer_call_fn_115744O3?0
)?&
 ?
inputs?????????

p
? "??????????
v
#__inference_h3_layer_call_fn_115767O'(/?,
%?"
 ?
inputs?????????

? "???????????
B__inference_DeepNN_layer_call_and_return_conditional_losses_115412l'(129?6
/?,
"?
h1_input?????????)
p 

 
? "%?"
?
0?????????
? v
#__inference_h1_layer_call_fn_115661O/?,
%?"
 ?
inputs?????????)
? "???????????
'__inference_DeepNN_layer_call_fn_115481_'(129?6
/?,
"?
h1_input?????????)
p 

 
? "???????????
A__inference_drop2_layer_call_and_return_conditional_losses_115739\3?0
)?&
 ?
inputs?????????

p 
? "%?"
?
0?????????

? ?
'__inference_DeepNN_layer_call_fn_115643]'(127?4
-?*
 ?
inputs?????????)
p 

 
? "???????????
'__inference_DeepNN_layer_call_fn_115446_'(129?6
/?,
"?
h1_input?????????)
p

 
? "???????????
B__inference_DeepNN_layer_call_and_return_conditional_losses_115617j'(127?4
-?*
 ?
inputs?????????)
p 

 
? "%?"
?
0?????????
? y
&__inference_drop3_layer_call_fn_115802O3?0
)?&
 ?
inputs?????????
p 
? "???????????
>__inference_h1_layer_call_and_return_conditional_losses_115654\/?,
%?"
 ?
inputs?????????)
? "%?"
?
0?????????
? 