╚П
ѓМ
D
AddV2
x"T
y"T
z"T"
Ttype:
2	ђљ
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( ѕ
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
Џ
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
,
Exp
x"T
y"T"
Ttype:

2
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
.
Identity

input"T
output"T"	
Ttype
\
	LeakyRelu
features"T
activations"T"
alphafloat%═╠L>"
Ttype0:
2
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(ѕ
?
Mul
x"T
y"T
z"T"
Ttype:
2	љ
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
Ё
RandomStandardNormal

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	ѕ
@
ReadVariableOp
resource
value"dtype"
dtypetypeѕ
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
list(type)(0ѕ
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
list(type)(0ѕ
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
┴
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
executor_typestring ѕе
@
StaticRegexFullMatch	
input

output
"
patternstring
Ш
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
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
ќ
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ѕ"serve*2.8.02v2.8.0-rc1-32-g3f878cff5b68уО
z
conv1d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv1d/kernel
s
!conv1d/kernel/Read/ReadVariableOpReadVariableOpconv1d/kernel*"
_output_shapes
: *
dtype0
n
conv1d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv1d/bias
g
conv1d/bias/Read/ReadVariableOpReadVariableOpconv1d/bias*
_output_shapes
: *
dtype0
і
batch_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: **
shared_namebatch_normalization/gamma
Ѓ
-batch_normalization/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization/gamma*
_output_shapes
: *
dtype0
ѕ
batch_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *)
shared_namebatch_normalization/beta
Ђ
,batch_normalization/beta/Read/ReadVariableOpReadVariableOpbatch_normalization/beta*
_output_shapes
: *
dtype0
ќ
batch_normalization/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *0
shared_name!batch_normalization/moving_mean
Ј
3batch_normalization/moving_mean/Read/ReadVariableOpReadVariableOpbatch_normalization/moving_mean*
_output_shapes
: *
dtype0
ъ
#batch_normalization/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *4
shared_name%#batch_normalization/moving_variance
Ќ
7batch_normalization/moving_variance/Read/ReadVariableOpReadVariableOp#batch_normalization/moving_variance*
_output_shapes
: *
dtype0
~
conv1d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @* 
shared_nameconv1d_1/kernel
w
#conv1d_1/kernel/Read/ReadVariableOpReadVariableOpconv1d_1/kernel*"
_output_shapes
: @*
dtype0
r
conv1d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv1d_1/bias
k
!conv1d_1/bias/Read/ReadVariableOpReadVariableOpconv1d_1/bias*
_output_shapes
:@*
dtype0
ј
batch_normalization_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatch_normalization_1/gamma
Є
/batch_normalization_1/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_1/gamma*
_output_shapes
:@*
dtype0
ї
batch_normalization_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_namebatch_normalization_1/beta
Ё
.batch_normalization_1/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_1/beta*
_output_shapes
:@*
dtype0
џ
!batch_normalization_1/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!batch_normalization_1/moving_mean
Њ
5batch_normalization_1/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_1/moving_mean*
_output_shapes
:@*
dtype0
б
%batch_normalization_1/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*6
shared_name'%batch_normalization_1/moving_variance
Џ
9batch_normalization_1/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_1/moving_variance*
_output_shapes
:@*
dtype0

conv1d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@ђ* 
shared_nameconv1d_2/kernel
x
#conv1d_2/kernel/Read/ReadVariableOpReadVariableOpconv1d_2/kernel*#
_output_shapes
:@ђ*
dtype0
s
conv1d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*
shared_nameconv1d_2/bias
l
!conv1d_2/bias/Read/ReadVariableOpReadVariableOpconv1d_2/bias*
_output_shapes	
:ђ*
dtype0
Ј
batch_normalization_2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*,
shared_namebatch_normalization_2/gamma
ѕ
/batch_normalization_2/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_2/gamma*
_output_shapes	
:ђ*
dtype0
Ї
batch_normalization_2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*+
shared_namebatch_normalization_2/beta
є
.batch_normalization_2/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_2/beta*
_output_shapes	
:ђ*
dtype0
Џ
!batch_normalization_2/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*2
shared_name#!batch_normalization_2/moving_mean
ћ
5batch_normalization_2/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_2/moving_mean*
_output_shapes	
:ђ*
dtype0
Б
%batch_normalization_2/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*6
shared_name'%batch_normalization_2/moving_variance
ю
9batch_normalization_2/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_2/moving_variance*
_output_shapes	
:ђ*
dtype0

z_log_mean/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ђ*"
shared_namez_log_mean/kernel
x
%z_log_mean/kernel/Read/ReadVariableOpReadVariableOpz_log_mean/kernel*
_output_shapes
:	ђ*
dtype0
v
z_log_mean/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_namez_log_mean/bias
o
#z_log_mean/bias/Read/ReadVariableOpReadVariableOpz_log_mean/bias*
_output_shapes
:*
dtype0
}
z_log_var/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ђ*!
shared_namez_log_var/kernel
v
$z_log_var/kernel/Read/ReadVariableOpReadVariableOpz_log_var/kernel*
_output_shapes
:	ђ*
dtype0
t
z_log_var/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namez_log_var/bias
m
"z_log_var/bias/Read/ReadVariableOpReadVariableOpz_log_var/bias*
_output_shapes
:*
dtype0

NoOpNoOp
тW
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*аW
valueќWBЊW BїW
В
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer-6
layer_with_weights-4
layer-7
	layer_with_weights-5
	layer-8

layer-9
layer-10
layer_with_weights-6
layer-11
layer_with_weights-7
layer-12
layer-13

signatures
#_self_saveable_object_factories
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature*
'
#_self_saveable_object_factories* 
╦

kernel
bias
#_self_saveable_object_factories
	variables
trainable_variables
regularization_losses
	keras_api
 __call__
*!&call_and_return_all_conditional_losses*
Щ
"axis
	#gamma
$beta
%moving_mean
&moving_variance
#'_self_saveable_object_factories
(	variables
)trainable_variables
*regularization_losses
+	keras_api
,__call__
*-&call_and_return_all_conditional_losses*
│
#._self_saveable_object_factories
/	variables
0trainable_variables
1regularization_losses
2	keras_api
3__call__
*4&call_and_return_all_conditional_losses* 
╦

5kernel
6bias
#7_self_saveable_object_factories
8	variables
9trainable_variables
:regularization_losses
;	keras_api
<__call__
*=&call_and_return_all_conditional_losses*
Щ
>axis
	?gamma
@beta
Amoving_mean
Bmoving_variance
#C_self_saveable_object_factories
D	variables
Etrainable_variables
Fregularization_losses
G	keras_api
H__call__
*I&call_and_return_all_conditional_losses*
│
#J_self_saveable_object_factories
K	variables
Ltrainable_variables
Mregularization_losses
N	keras_api
O__call__
*P&call_and_return_all_conditional_losses* 
╦

Qkernel
Rbias
#S_self_saveable_object_factories
T	variables
Utrainable_variables
Vregularization_losses
W	keras_api
X__call__
*Y&call_and_return_all_conditional_losses*
Щ
Zaxis
	[gamma
\beta
]moving_mean
^moving_variance
#__self_saveable_object_factories
`	variables
atrainable_variables
bregularization_losses
c	keras_api
d__call__
*e&call_and_return_all_conditional_losses*
│
#f_self_saveable_object_factories
g	variables
htrainable_variables
iregularization_losses
j	keras_api
k__call__
*l&call_and_return_all_conditional_losses* 
│
#m_self_saveable_object_factories
n	variables
otrainable_variables
pregularization_losses
q	keras_api
r__call__
*s&call_and_return_all_conditional_losses* 
╦

tkernel
ubias
#v_self_saveable_object_factories
w	variables
xtrainable_variables
yregularization_losses
z	keras_api
{__call__
*|&call_and_return_all_conditional_losses*
Л

}kernel
~bias
#_self_saveable_object_factories
ђ	variables
Ђtrainable_variables
ѓregularization_losses
Ѓ	keras_api
ё__call__
+Ё&call_and_return_all_conditional_losses*
║
$є_self_saveable_object_factories
Є	variables
ѕtrainable_variables
Ѕregularization_losses
і	keras_api
І__call__
+ї&call_and_return_all_conditional_losses* 

Їserving_default* 
* 
ф
0
1
#2
$3
%4
&5
56
67
?8
@9
A10
B11
Q12
R13
[14
\15
]16
^17
t18
u19
}20
~21*
z
0
1
#2
$3
54
65
?6
@7
Q8
R9
[10
\11
t12
u13
}14
~15*
* 
х
јnon_trainable_variables
Јlayers
љmetrics
 Љlayer_regularization_losses
њlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 
* 
]W
VARIABLE_VALUEconv1d/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEconv1d/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

0
1*

0
1*
* 
ў
Њnon_trainable_variables
ћlayers
Ћmetrics
 ќlayer_regularization_losses
Ќlayer_metrics
	variables
trainable_variables
regularization_losses
 __call__
*!&call_and_return_all_conditional_losses
&!"call_and_return_conditional_losses*
* 
* 
* 
hb
VARIABLE_VALUEbatch_normalization/gamma5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUEbatch_normalization/beta4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUEbatch_normalization/moving_mean;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUE#batch_normalization/moving_variance?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
* 
 
#0
$1
%2
&3*

#0
$1*
* 
ў
ўnon_trainable_variables
Ўlayers
џmetrics
 Џlayer_regularization_losses
юlayer_metrics
(	variables
)trainable_variables
*regularization_losses
,__call__
*-&call_and_return_all_conditional_losses
&-"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
ќ
Юnon_trainable_variables
ъlayers
Ъmetrics
 аlayer_regularization_losses
Аlayer_metrics
/	variables
0trainable_variables
1regularization_losses
3__call__
*4&call_and_return_all_conditional_losses
&4"call_and_return_conditional_losses* 
* 
* 
_Y
VARIABLE_VALUEconv1d_1/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEconv1d_1/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

50
61*

50
61*
* 
ў
бnon_trainable_variables
Бlayers
цmetrics
 Цlayer_regularization_losses
дlayer_metrics
8	variables
9trainable_variables
:regularization_losses
<__call__
*=&call_and_return_all_conditional_losses
&="call_and_return_conditional_losses*
* 
* 
* 
jd
VARIABLE_VALUEbatch_normalization_1/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUEbatch_normalization_1/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUE!batch_normalization_1/moving_mean;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE%batch_normalization_1/moving_variance?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
* 
 
?0
@1
A2
B3*

?0
@1*
* 
ў
Дnon_trainable_variables
еlayers
Еmetrics
 фlayer_regularization_losses
Фlayer_metrics
D	variables
Etrainable_variables
Fregularization_losses
H__call__
*I&call_and_return_all_conditional_losses
&I"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
ќ
гnon_trainable_variables
Гlayers
«metrics
 »layer_regularization_losses
░layer_metrics
K	variables
Ltrainable_variables
Mregularization_losses
O__call__
*P&call_and_return_all_conditional_losses
&P"call_and_return_conditional_losses* 
* 
* 
_Y
VARIABLE_VALUEconv1d_2/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEconv1d_2/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

Q0
R1*

Q0
R1*
* 
ў
▒non_trainable_variables
▓layers
│metrics
 ┤layer_regularization_losses
хlayer_metrics
T	variables
Utrainable_variables
Vregularization_losses
X__call__
*Y&call_and_return_all_conditional_losses
&Y"call_and_return_conditional_losses*
* 
* 
* 
jd
VARIABLE_VALUEbatch_normalization_2/gamma5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUEbatch_normalization_2/beta4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUE!batch_normalization_2/moving_mean;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE%batch_normalization_2/moving_variance?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
* 
 
[0
\1
]2
^3*

[0
\1*
* 
ў
Хnon_trainable_variables
иlayers
Иmetrics
 ╣layer_regularization_losses
║layer_metrics
`	variables
atrainable_variables
bregularization_losses
d__call__
*e&call_and_return_all_conditional_losses
&e"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
ќ
╗non_trainable_variables
╝layers
йmetrics
 Йlayer_regularization_losses
┐layer_metrics
g	variables
htrainable_variables
iregularization_losses
k__call__
*l&call_and_return_all_conditional_losses
&l"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
* 
ќ
└non_trainable_variables
┴layers
┬metrics
 ├layer_regularization_losses
─layer_metrics
n	variables
otrainable_variables
pregularization_losses
r__call__
*s&call_and_return_all_conditional_losses
&s"call_and_return_conditional_losses* 
* 
* 
a[
VARIABLE_VALUEz_log_mean/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEz_log_mean/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

t0
u1*

t0
u1*
* 
ў
┼non_trainable_variables
кlayers
Кmetrics
 ╚layer_regularization_losses
╔layer_metrics
w	variables
xtrainable_variables
yregularization_losses
{__call__
*|&call_and_return_all_conditional_losses
&|"call_and_return_conditional_losses*
* 
* 
`Z
VARIABLE_VALUEz_log_var/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEz_log_var/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

}0
~1*

}0
~1*
* 
ъ
╩non_trainable_variables
╦layers
╠metrics
 ═layer_regularization_losses
╬layer_metrics
ђ	variables
Ђtrainable_variables
ѓregularization_losses
ё__call__
+Ё&call_and_return_all_conditional_losses
'Ё"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
ю
¤non_trainable_variables
лlayers
Лmetrics
 мlayer_regularization_losses
Мlayer_metrics
Є	variables
ѕtrainable_variables
Ѕregularization_losses
І__call__
+ї&call_and_return_all_conditional_losses
'ї"call_and_return_conditional_losses* 
* 
* 
* 
.
%0
&1
A2
B3
]4
^5*
j
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
13*
* 
* 
* 
* 
* 
* 
* 
* 

%0
&1*
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
A0
B1*
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
]0
^1*
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
ѓ
serving_default_input_1Placeholder*+
_output_shapes
:         *
dtype0* 
shape:         
Е
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1conv1d/kernelconv1d/bias#batch_normalization/moving_variancebatch_normalization/gammabatch_normalization/moving_meanbatch_normalization/betaconv1d_1/kernelconv1d_1/bias%batch_normalization_1/moving_variancebatch_normalization_1/gamma!batch_normalization_1/moving_meanbatch_normalization_1/betaconv1d_2/kernelconv1d_2/bias%batch_normalization_2/moving_variancebatch_normalization_2/gamma!batch_normalization_2/moving_meanbatch_normalization_2/betaz_log_mean/kernelz_log_mean/biasz_log_var/kernelz_log_var/bias*"
Tin
2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:         :         :         *8
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8ѓ *-
f(R&
$__inference_signature_wrapper_119739
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
ї

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename!conv1d/kernel/Read/ReadVariableOpconv1d/bias/Read/ReadVariableOp-batch_normalization/gamma/Read/ReadVariableOp,batch_normalization/beta/Read/ReadVariableOp3batch_normalization/moving_mean/Read/ReadVariableOp7batch_normalization/moving_variance/Read/ReadVariableOp#conv1d_1/kernel/Read/ReadVariableOp!conv1d_1/bias/Read/ReadVariableOp/batch_normalization_1/gamma/Read/ReadVariableOp.batch_normalization_1/beta/Read/ReadVariableOp5batch_normalization_1/moving_mean/Read/ReadVariableOp9batch_normalization_1/moving_variance/Read/ReadVariableOp#conv1d_2/kernel/Read/ReadVariableOp!conv1d_2/bias/Read/ReadVariableOp/batch_normalization_2/gamma/Read/ReadVariableOp.batch_normalization_2/beta/Read/ReadVariableOp5batch_normalization_2/moving_mean/Read/ReadVariableOp9batch_normalization_2/moving_variance/Read/ReadVariableOp%z_log_mean/kernel/Read/ReadVariableOp#z_log_mean/bias/Read/ReadVariableOp$z_log_var/kernel/Read/ReadVariableOp"z_log_var/bias/Read/ReadVariableOpConst*#
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *(
f#R!
__inference__traced_save_120221
¤
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv1d/kernelconv1d/biasbatch_normalization/gammabatch_normalization/betabatch_normalization/moving_mean#batch_normalization/moving_varianceconv1d_1/kernelconv1d_1/biasbatch_normalization_1/gammabatch_normalization_1/beta!batch_normalization_1/moving_mean%batch_normalization_1/moving_varianceconv1d_2/kernelconv1d_2/biasbatch_normalization_2/gammabatch_normalization_2/beta!batch_normalization_2/moving_mean%batch_normalization_2/moving_variancez_log_mean/kernelz_log_mean/biasz_log_var/kernelz_log_var/bias*"
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *+
f&R$
"__inference__traced_restore_120297ри
з
Н
$__inference_signature_wrapper_119739
input_1
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: 
	unknown_5: @
	unknown_6:@
	unknown_7:@
	unknown_8:@
	unknown_9:@

unknown_10:@!

unknown_11:@ђ

unknown_12:	ђ

unknown_13:	ђ

unknown_14:	ђ

unknown_15:	ђ

unknown_16:	ђ

unknown_17:	ђ

unknown_18:

unknown_19:	ђ

unknown_20:
identity

identity_1

identity_2ѕбStatefulPartitionedCallЬ
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_20*"
Tin
2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:         :         :         *8
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8ѓ **
f%R#
!__inference__wrapped_model_118445o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:         q

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:         : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:         
!
_user_specified_name	input_1
▄
Л
6__inference_batch_normalization_1_layer_call_fn_119890

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityѕбStatefulPartitionedCallЇ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  @*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *Z
fURS
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_118551|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :                  @`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:                  @: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :                  @
 
_user_specified_nameinputs
№
Њ
D__inference_conv1d_1_layer_call_and_return_conditional_losses_118750

inputsA
+conv1d_expanddims_1_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identityѕбBiasAdd/ReadVariableOpб"Conv1D/ExpandDims_1/ReadVariableOp`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§        Ђ
Conv1D/ExpandDims
ExpandDimsinputsConv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:          њ
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: @*
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : а
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: @г
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
ђ
Conv1D/SqueezeSqueezeConv1D:output:0*
T0*+
_output_shapes
:         @*
squeeze_dims

§        r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ђ
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         @c
IdentityIdentityBiasAdd:output:0^NoOp*
T0*+
_output_shapes
:         @ё
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:          : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:          
 
_user_specified_nameinputs
┐
_
C__inference_flatten_layer_call_and_return_conditional_losses_120092

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"       ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:         ђY
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:         ђ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ђ:T P
,
_output_shapes
:         ђ
 
_user_specified_nameinputs
Ј
░
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_118551

inputs/
!batchnorm_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@1
#batchnorm_readvariableop_1_resource:@1
#batchnorm_readvariableop_2_resource:@
identityѕбbatchnorm/ReadVariableOpбbatchnorm/ReadVariableOp_1бbatchnorm/ReadVariableOp_2бbatchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@p
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :                  @z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:@z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :                  @o
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :                  @║
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:                  @: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :                  @
 
_user_specified_nameinputs
Ї
«
O__inference_batch_normalization_layer_call_and_return_conditional_losses_119809

inputs/
!batchnorm_readvariableop_resource: 3
%batchnorm_mul_readvariableop_resource: 1
#batchnorm_readvariableop_1_resource: 1
#batchnorm_readvariableop_2_resource: 
identityѕбbatchnorm/ReadVariableOpбbatchnorm/ReadVariableOp_1бbatchnorm/ReadVariableOp_2бbatchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:w
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
: p
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :                   z
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
: 
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :                   o
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :                   ║
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:                   : : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :                   
 
_user_specified_nameinputs
═	
Э
F__inference_z_log_mean_layer_call_and_return_conditional_losses_120111

inputs1
matmul_readvariableop_resource:	ђ-
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	ђ*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ђ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
ч%
У
O__inference_batch_normalization_layer_call_and_return_conditional_losses_119843

inputs5
'assignmovingavg_readvariableop_resource: 7
)assignmovingavg_1_readvariableop_resource: 3
%batchnorm_mul_readvariableop_resource: /
!batchnorm_readvariableop_resource: 
identityѕбAssignMovingAvgбAssignMovingAvg/ReadVariableOpбAssignMovingAvg_1б AssignMovingAvg_1/ReadVariableOpбbatchnorm/ReadVariableOpбbatchnorm/mul/ReadVariableOpo
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       Ѓ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
: *
	keep_dims(h
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
: ћ
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*4
_output_shapes"
 :                   s
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       б
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
: *
	keep_dims(n
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 t
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<ѓ
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
: *
dtype0Ђ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
: x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
: г
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
О#<є
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
: *
dtype0Є
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
: ~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
: ┤
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:q
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
: p
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :                   h
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
: 
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :                   o
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :                   Ж
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:                   : : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :                   
 
_user_specified_nameinputs
є
s
D__inference_sampling_layer_call_and_return_conditional_losses_114491
inputs_0
inputs_1
identityѕ=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Л
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
Shape_1Shapeinputs_0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask{
random_normal/shapePackstrided_slice:output:0strided_slice_1:output:0*
N*
T0*
_output_shapes
:W
random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    Y
random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?х
"random_normal/RandomStandardNormalRandomStandardNormalrandom_normal/shape:output:0*
T0*'
_output_shapes
:         *
dtype0*
seed▒ т)*
seed2Ћу2ќ
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*'
_output_shapes
:         |
random_normalAddV2random_normal/mul:z:0random_normal/mean:output:0*
T0*'
_output_shapes
:         J
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?V
mulMulmul/x:output:0inputs_1*
T0*'
_output_shapes
:         E
ExpExpmul:z:0*
T0*'
_output_shapes
:         Z
mul_1MulExp:y:0random_normal:z:0*
T0*'
_output_shapes
:         S
addAddV2inputs_0	mul_1:z:0*
T0*'
_output_shapes
:         O
IdentityIdentityadd:z:0*
T0*'
_output_shapes
:         "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:         :         :Q M
'
_output_shapes
:         
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:         
"
_user_specified_name
inputs/1
м
ў
'__inference_conv1d_layer_call_fn_119748

inputs
unknown: 
	unknown_0: 
identityѕбStatefulPartitionedCall█
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:          *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *K
fFRD
B__inference_conv1d_layer_call_and_return_conditional_losses_118713s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:          `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
п
¤
4__inference_batch_normalization_layer_call_fn_119776

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identityѕбStatefulPartitionedCallІ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                   *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_118469|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :                   `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:                   : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :                   
 
_user_specified_nameinputs
з
e
I__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_118770

inputs
identity[
	LeakyRelu	LeakyReluinputs*+
_output_shapes
:         @*
alpha%џЎЎ>c
IdentityIdentityLeakyRelu:activations:0*
T0*+
_output_shapes
:         @"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         @:S O
+
_output_shapes
:         @
 
_user_specified_nameinputs
їC
╦

C__inference_encoder_layer_call_and_return_conditional_losses_118853

inputs#
conv1d_118714: 
conv1d_118716: (
batch_normalization_118719: (
batch_normalization_118721: (
batch_normalization_118723: (
batch_normalization_118725: %
conv1d_1_118751: @
conv1d_1_118753:@*
batch_normalization_1_118756:@*
batch_normalization_1_118758:@*
batch_normalization_1_118760:@*
batch_normalization_1_118762:@&
conv1d_2_118788:@ђ
conv1d_2_118790:	ђ+
batch_normalization_2_118793:	ђ+
batch_normalization_2_118795:	ђ+
batch_normalization_2_118797:	ђ+
batch_normalization_2_118799:	ђ$
z_log_mean_118828:	ђ
z_log_mean_118830:#
z_log_var_118844:	ђ
z_log_var_118846:
identity

identity_1

identity_2ѕб+batch_normalization/StatefulPartitionedCallб-batch_normalization_1/StatefulPartitionedCallб-batch_normalization_2/StatefulPartitionedCallбconv1d/StatefulPartitionedCallб conv1d_1/StatefulPartitionedCallб conv1d_2/StatefulPartitionedCallб sampling/StatefulPartitionedCallб"z_log_mean/StatefulPartitionedCallб!z_log_var/StatefulPartitionedCallВ
conv1d/StatefulPartitionedCallStatefulPartitionedCallinputsconv1d_118714conv1d_118716*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:          *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *K
fFRD
B__inference_conv1d_layer_call_and_return_conditional_losses_118713§
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall'conv1d/StatefulPartitionedCall:output:0batch_normalization_118719batch_normalization_118721batch_normalization_118723batch_normalization_118725*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:          *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_118469­
leaky_re_lu/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:          * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *P
fKRI
G__inference_leaky_re_lu_layer_call_and_return_conditional_losses_118733њ
 conv1d_1/StatefulPartitionedCallStatefulPartitionedCall$leaky_re_lu/PartitionedCall:output:0conv1d_1_118751conv1d_1_118753*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_conv1d_1_layer_call_and_return_conditional_losses_118750І
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall)conv1d_1/StatefulPartitionedCall:output:0batch_normalization_1_118756batch_normalization_1_118758batch_normalization_1_118760batch_normalization_1_118762*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         @*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *Z
fURS
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_118551Ш
leaky_re_lu_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *R
fMRK
I__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_118770Ћ
 conv1d_2/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_1/PartitionedCall:output:0conv1d_2_118788conv1d_2_118790*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ђ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_conv1d_2_layer_call_and_return_conditional_losses_118787ї
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall)conv1d_2/StatefulPartitionedCall:output:0batch_normalization_2_118793batch_normalization_2_118795batch_normalization_2_118797batch_normalization_2_118799*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ђ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *Z
fURS
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_118633э
leaky_re_lu_2/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *R
fMRK
I__inference_leaky_re_lu_2_layer_call_and_return_conditional_losses_118807О
flatten/PartitionedCallPartitionedCall&leaky_re_lu_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_118815њ
"z_log_mean/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0z_log_mean_118828z_log_mean_118830*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *O
fJRH
F__inference_z_log_mean_layer_call_and_return_conditional_losses_118827ј
!z_log_var/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0z_log_var_118844z_log_var_118846*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *N
fIRG
E__inference_z_log_var_layer_call_and_return_conditional_losses_118843 
 sampling/StatefulPartitionedCallStatefulPartitionedCall+z_log_mean/StatefulPartitionedCall:output:0*z_log_var/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *2
f-R+
)__inference_restored_function_body_118440z
IdentityIdentity+z_log_mean/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         {

Identity_1Identity*z_log_var/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         z

Identity_2Identity)sampling/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         Д
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall^conv1d/StatefulPartitionedCall!^conv1d_1/StatefulPartitionedCall!^conv1d_2/StatefulPartitionedCall!^sampling/StatefulPartitionedCall#^z_log_mean/StatefulPartitionedCall"^z_log_var/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:         : : : : : : : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2@
conv1d/StatefulPartitionedCallconv1d/StatefulPartitionedCall2D
 conv1d_1/StatefulPartitionedCall conv1d_1/StatefulPartitionedCall2D
 conv1d_2/StatefulPartitionedCall conv1d_2/StatefulPartitionedCall2D
 sampling/StatefulPartitionedCall sampling/StatefulPartitionedCall2H
"z_log_mean/StatefulPartitionedCall"z_log_mean/StatefulPartitionedCall2F
!z_log_var/StatefulPartitionedCall!z_log_var/StatefulPartitionedCall:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
б
┤
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_118633

inputs0
!batchnorm_readvariableop_resource:	ђ4
%batchnorm_mul_readvariableop_resource:	ђ2
#batchnorm_readvariableop_1_resource:	ђ2
#batchnorm_readvariableop_2_resource:	ђ
identityѕбbatchnorm/ReadVariableOpбbatchnorm/ReadVariableOp_1бbatchnorm/ReadVariableOp_2бbatchnorm/mul/ReadVariableOpw
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:ђ*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:x
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:ђQ
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:ђ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:ђ*
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:ђq
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*5
_output_shapes#
!:                  ђ{
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0s
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:ђ{
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:ђ*
dtype0s
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:ђђ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*5
_output_shapes#
!:                  ђp
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*5
_output_shapes#
!:                  ђ║
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):                  ђ: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:] Y
5
_output_shapes#
!:                  ђ
 
_user_specified_nameinputs
Ї
«
O__inference_batch_normalization_layer_call_and_return_conditional_losses_118469

inputs/
!batchnorm_readvariableop_resource: 3
%batchnorm_mul_readvariableop_resource: 1
#batchnorm_readvariableop_1_resource: 1
#batchnorm_readvariableop_2_resource: 
identityѕбbatchnorm/ReadVariableOpбbatchnorm/ReadVariableOp_1бbatchnorm/ReadVariableOp_2бbatchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:w
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
: p
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :                   z
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
: 
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :                   o
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :                   ║
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:                   : : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :                   
 
_user_specified_nameinputs
Њ
┘
(__inference_encoder_layer_call_fn_119200
input_1
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: 
	unknown_5: @
	unknown_6:@
	unknown_7:@
	unknown_8:@
	unknown_9:@

unknown_10:@!

unknown_11:@ђ

unknown_12:	ђ

unknown_13:	ђ

unknown_14:	ђ

unknown_15:	ђ

unknown_16:	ђ

unknown_17:	ђ

unknown_18:

unknown_19:	ђ

unknown_20:
identity

identity_1

identity_2ѕбStatefulPartitionedCallі
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_20*"
Tin
2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:         :         :         *2
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_encoder_layer_call_and_return_conditional_losses_119096o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:         q

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:         : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:         
!
_user_specified_name	input_1
ь
Љ
B__inference_conv1d_layer_call_and_return_conditional_losses_118713

inputsA
+conv1d_expanddims_1_readvariableop_resource: -
biasadd_readvariableop_resource: 
identityѕбBiasAdd/ReadVariableOpб"Conv1D/ExpandDims_1/ReadVariableOp`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§        Ђ
Conv1D/ExpandDims
ExpandDimsinputsConv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:         њ
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : а
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: г
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:          *
paddingSAME*
strides
ђ
Conv1D/SqueezeSqueezeConv1D:output:0*
T0*+
_output_shapes
:          *
squeeze_dims

§        r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0Ђ
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:          c
IdentityIdentityBiasAdd:output:0^NoOp*
T0*+
_output_shapes
:          ё
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
═
p
)__inference_restored_function_body_118440

inputs
inputs_1
identityѕбStatefulPartitionedCallФ
StatefulPartitionedCallStatefulPartitionedCallinputsinputs_1*
Tin
2*
Tout
2*'
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_sampling_layer_call_and_return_conditional_losses_114491o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:         :         22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs:OK
'
_output_shapes
:         
 
_user_specified_nameinputs
ЅC
╠

C__inference_encoder_layer_call_and_return_conditional_losses_119326
input_1#
conv1d_119266: 
conv1d_119268: (
batch_normalization_119271: (
batch_normalization_119273: (
batch_normalization_119275: (
batch_normalization_119277: %
conv1d_1_119281: @
conv1d_1_119283:@*
batch_normalization_1_119286:@*
batch_normalization_1_119288:@*
batch_normalization_1_119290:@*
batch_normalization_1_119292:@&
conv1d_2_119296:@ђ
conv1d_2_119298:	ђ+
batch_normalization_2_119301:	ђ+
batch_normalization_2_119303:	ђ+
batch_normalization_2_119305:	ђ+
batch_normalization_2_119307:	ђ$
z_log_mean_119312:	ђ
z_log_mean_119314:#
z_log_var_119317:	ђ
z_log_var_119319:
identity

identity_1

identity_2ѕб+batch_normalization/StatefulPartitionedCallб-batch_normalization_1/StatefulPartitionedCallб-batch_normalization_2/StatefulPartitionedCallбconv1d/StatefulPartitionedCallб conv1d_1/StatefulPartitionedCallб conv1d_2/StatefulPartitionedCallб sampling/StatefulPartitionedCallб"z_log_mean/StatefulPartitionedCallб!z_log_var/StatefulPartitionedCallь
conv1d/StatefulPartitionedCallStatefulPartitionedCallinput_1conv1d_119266conv1d_119268*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:          *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *K
fFRD
B__inference_conv1d_layer_call_and_return_conditional_losses_118713ч
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall'conv1d/StatefulPartitionedCall:output:0batch_normalization_119271batch_normalization_119273batch_normalization_119275batch_normalization_119277*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:          *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_118516­
leaky_re_lu/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:          * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *P
fKRI
G__inference_leaky_re_lu_layer_call_and_return_conditional_losses_118733њ
 conv1d_1/StatefulPartitionedCallStatefulPartitionedCall$leaky_re_lu/PartitionedCall:output:0conv1d_1_119281conv1d_1_119283*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_conv1d_1_layer_call_and_return_conditional_losses_118750Ѕ
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall)conv1d_1/StatefulPartitionedCall:output:0batch_normalization_1_119286batch_normalization_1_119288batch_normalization_1_119290batch_normalization_1_119292*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *Z
fURS
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_118598Ш
leaky_re_lu_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *R
fMRK
I__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_118770Ћ
 conv1d_2/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_1/PartitionedCall:output:0conv1d_2_119296conv1d_2_119298*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ђ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_conv1d_2_layer_call_and_return_conditional_losses_118787і
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall)conv1d_2/StatefulPartitionedCall:output:0batch_normalization_2_119301batch_normalization_2_119303batch_normalization_2_119305batch_normalization_2_119307*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ђ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *Z
fURS
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_118680э
leaky_re_lu_2/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *R
fMRK
I__inference_leaky_re_lu_2_layer_call_and_return_conditional_losses_118807О
flatten/PartitionedCallPartitionedCall&leaky_re_lu_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_118815њ
"z_log_mean/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0z_log_mean_119312z_log_mean_119314*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *O
fJRH
F__inference_z_log_mean_layer_call_and_return_conditional_losses_118827ј
!z_log_var/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0z_log_var_119317z_log_var_119319*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *N
fIRG
E__inference_z_log_var_layer_call_and_return_conditional_losses_118843 
 sampling/StatefulPartitionedCallStatefulPartitionedCall+z_log_mean/StatefulPartitionedCall:output:0*z_log_var/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *2
f-R+
)__inference_restored_function_body_118440z
IdentityIdentity+z_log_mean/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         {

Identity_1Identity*z_log_var/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         z

Identity_2Identity)sampling/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         Д
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall^conv1d/StatefulPartitionedCall!^conv1d_1/StatefulPartitionedCall!^conv1d_2/StatefulPartitionedCall!^sampling/StatefulPartitionedCall#^z_log_mean/StatefulPartitionedCall"^z_log_var/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:         : : : : : : : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2@
conv1d/StatefulPartitionedCallconv1d/StatefulPartitionedCall2D
 conv1d_1/StatefulPartitionedCall conv1d_1/StatefulPartitionedCall2D
 conv1d_2/StatefulPartitionedCall conv1d_2/StatefulPartitionedCall2D
 sampling/StatefulPartitionedCall sampling/StatefulPartitionedCall2H
"z_log_mean/StatefulPartitionedCall"z_log_mean/StatefulPartitionedCall2F
!z_log_var/StatefulPartitionedCall!z_log_var/StatefulPartitionedCall:T P
+
_output_shapes
:         
!
_user_specified_name	input_1
о
¤
4__inference_batch_normalization_layer_call_fn_119789

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identityѕбStatefulPartitionedCallЅ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                   *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_118516|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :                   `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:                   : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :                   
 
_user_specified_nameinputs
э
e
I__inference_leaky_re_lu_2_layer_call_and_return_conditional_losses_120081

inputs
identity\
	LeakyRelu	LeakyReluinputs*,
_output_shapes
:         ђ*
alpha%џЎЎ>d
IdentityIdentityLeakyRelu:activations:0*
T0*,
_output_shapes
:         ђ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ђ:T P
,
_output_shapes
:         ђ
 
_user_specified_nameinputs
┴5
╣

__inference__traced_save_120221
file_prefix,
(savev2_conv1d_kernel_read_readvariableop*
&savev2_conv1d_bias_read_readvariableop8
4savev2_batch_normalization_gamma_read_readvariableop7
3savev2_batch_normalization_beta_read_readvariableop>
:savev2_batch_normalization_moving_mean_read_readvariableopB
>savev2_batch_normalization_moving_variance_read_readvariableop.
*savev2_conv1d_1_kernel_read_readvariableop,
(savev2_conv1d_1_bias_read_readvariableop:
6savev2_batch_normalization_1_gamma_read_readvariableop9
5savev2_batch_normalization_1_beta_read_readvariableop@
<savev2_batch_normalization_1_moving_mean_read_readvariableopD
@savev2_batch_normalization_1_moving_variance_read_readvariableop.
*savev2_conv1d_2_kernel_read_readvariableop,
(savev2_conv1d_2_bias_read_readvariableop:
6savev2_batch_normalization_2_gamma_read_readvariableop9
5savev2_batch_normalization_2_beta_read_readvariableop@
<savev2_batch_normalization_2_moving_mean_read_readvariableopD
@savev2_batch_normalization_2_moving_variance_read_readvariableop0
,savev2_z_log_mean_kernel_read_readvariableop.
*savev2_z_log_mean_bias_read_readvariableop/
+savev2_z_log_var_kernel_read_readvariableop-
)savev2_z_log_var_bias_read_readvariableop
savev2_const

identity_1ѕбMergeV2Checkpointsw
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
_temp/partЂ
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
value	B : Њ
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ы

SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*џ

valueљ
BЇ
B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHЏ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*A
value8B6B B B B B B B B B B B B B B B B B B B B B B B ╗

SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0(savev2_conv1d_kernel_read_readvariableop&savev2_conv1d_bias_read_readvariableop4savev2_batch_normalization_gamma_read_readvariableop3savev2_batch_normalization_beta_read_readvariableop:savev2_batch_normalization_moving_mean_read_readvariableop>savev2_batch_normalization_moving_variance_read_readvariableop*savev2_conv1d_1_kernel_read_readvariableop(savev2_conv1d_1_bias_read_readvariableop6savev2_batch_normalization_1_gamma_read_readvariableop5savev2_batch_normalization_1_beta_read_readvariableop<savev2_batch_normalization_1_moving_mean_read_readvariableop@savev2_batch_normalization_1_moving_variance_read_readvariableop*savev2_conv1d_2_kernel_read_readvariableop(savev2_conv1d_2_bias_read_readvariableop6savev2_batch_normalization_2_gamma_read_readvariableop5savev2_batch_normalization_2_beta_read_readvariableop<savev2_batch_normalization_2_moving_mean_read_readvariableop@savev2_batch_normalization_2_moving_variance_read_readvariableop,savev2_z_log_mean_kernel_read_readvariableop*savev2_z_log_mean_bias_read_readvariableop+savev2_z_log_var_kernel_read_readvariableop)savev2_z_log_var_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *%
dtypes
2љ
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:І
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

identity_1Identity_1:output:0*┼
_input_shapes│
░: : : : : : : : @:@:@:@:@:@:@ђ:ђ:ђ:ђ:ђ:ђ:	ђ::	ђ:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:($
"
_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: :($
"
_output_shapes
: @: 

_output_shapes
:@: 	

_output_shapes
:@: 


_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@:)%
#
_output_shapes
:@ђ:!

_output_shapes	
:ђ:!

_output_shapes	
:ђ:!

_output_shapes	
:ђ:!

_output_shapes	
:ђ:!

_output_shapes	
:ђ:%!

_output_shapes
:	ђ: 

_output_shapes
::%!

_output_shapes
:	ђ: 

_output_shapes
::

_output_shapes
: 
ЗЉ
┘
C__inference_encoder_layer_call_and_return_conditional_losses_119537

inputsH
2conv1d_conv1d_expanddims_1_readvariableop_resource: 4
&conv1d_biasadd_readvariableop_resource: C
5batch_normalization_batchnorm_readvariableop_resource: G
9batch_normalization_batchnorm_mul_readvariableop_resource: E
7batch_normalization_batchnorm_readvariableop_1_resource: E
7batch_normalization_batchnorm_readvariableop_2_resource: J
4conv1d_1_conv1d_expanddims_1_readvariableop_resource: @6
(conv1d_1_biasadd_readvariableop_resource:@E
7batch_normalization_1_batchnorm_readvariableop_resource:@I
;batch_normalization_1_batchnorm_mul_readvariableop_resource:@G
9batch_normalization_1_batchnorm_readvariableop_1_resource:@G
9batch_normalization_1_batchnorm_readvariableop_2_resource:@K
4conv1d_2_conv1d_expanddims_1_readvariableop_resource:@ђ7
(conv1d_2_biasadd_readvariableop_resource:	ђF
7batch_normalization_2_batchnorm_readvariableop_resource:	ђJ
;batch_normalization_2_batchnorm_mul_readvariableop_resource:	ђH
9batch_normalization_2_batchnorm_readvariableop_1_resource:	ђH
9batch_normalization_2_batchnorm_readvariableop_2_resource:	ђ<
)z_log_mean_matmul_readvariableop_resource:	ђ8
*z_log_mean_biasadd_readvariableop_resource:;
(z_log_var_matmul_readvariableop_resource:	ђ7
)z_log_var_biasadd_readvariableop_resource:
identity

identity_1

identity_2ѕб,batch_normalization/batchnorm/ReadVariableOpб.batch_normalization/batchnorm/ReadVariableOp_1б.batch_normalization/batchnorm/ReadVariableOp_2б0batch_normalization/batchnorm/mul/ReadVariableOpб.batch_normalization_1/batchnorm/ReadVariableOpб0batch_normalization_1/batchnorm/ReadVariableOp_1б0batch_normalization_1/batchnorm/ReadVariableOp_2б2batch_normalization_1/batchnorm/mul/ReadVariableOpб.batch_normalization_2/batchnorm/ReadVariableOpб0batch_normalization_2/batchnorm/ReadVariableOp_1б0batch_normalization_2/batchnorm/ReadVariableOp_2б2batch_normalization_2/batchnorm/mul/ReadVariableOpбconv1d/BiasAdd/ReadVariableOpб)conv1d/Conv1D/ExpandDims_1/ReadVariableOpбconv1d_1/BiasAdd/ReadVariableOpб+conv1d_1/Conv1D/ExpandDims_1/ReadVariableOpбconv1d_2/BiasAdd/ReadVariableOpб+conv1d_2/Conv1D/ExpandDims_1/ReadVariableOpб sampling/StatefulPartitionedCallб!z_log_mean/BiasAdd/ReadVariableOpб z_log_mean/MatMul/ReadVariableOpб z_log_var/BiasAdd/ReadVariableOpбz_log_var/MatMul/ReadVariableOpg
conv1d/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§        Ј
conv1d/Conv1D/ExpandDims
ExpandDimsinputs%conv1d/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:         а
)conv1d/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp2conv1d_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype0`
conv1d/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : х
conv1d/Conv1D/ExpandDims_1
ExpandDims1conv1d/Conv1D/ExpandDims_1/ReadVariableOp:value:0'conv1d/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: ┴
conv1d/Conv1DConv2D!conv1d/Conv1D/ExpandDims:output:0#conv1d/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:          *
paddingSAME*
strides
ј
conv1d/Conv1D/SqueezeSqueezeconv1d/Conv1D:output:0*
T0*+
_output_shapes
:          *
squeeze_dims

§        ђ
conv1d/BiasAdd/ReadVariableOpReadVariableOp&conv1d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0ќ
conv1d/BiasAddBiasAddconv1d/Conv1D/Squeeze:output:0%conv1d/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:          ъ
,batch_normalization/batchnorm/ReadVariableOpReadVariableOp5batch_normalization_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype0h
#batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:│
!batch_normalization/batchnorm/addAddV24batch_normalization/batchnorm/ReadVariableOp:value:0,batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes
: x
#batch_normalization/batchnorm/RsqrtRsqrt%batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes
: д
0batch_normalization/batchnorm/mul/ReadVariableOpReadVariableOp9batch_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype0░
!batch_normalization/batchnorm/mulMul'batch_normalization/batchnorm/Rsqrt:y:08batch_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: а
#batch_normalization/batchnorm/mul_1Mulconv1d/BiasAdd:output:0%batch_normalization/batchnorm/mul:z:0*
T0*+
_output_shapes
:          б
.batch_normalization/batchnorm/ReadVariableOp_1ReadVariableOp7batch_normalization_batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype0«
#batch_normalization/batchnorm/mul_2Mul6batch_normalization/batchnorm/ReadVariableOp_1:value:0%batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes
: б
.batch_normalization/batchnorm/ReadVariableOp_2ReadVariableOp7batch_normalization_batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype0«
!batch_normalization/batchnorm/subSub6batch_normalization/batchnorm/ReadVariableOp_2:value:0'batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes
: ▓
#batch_normalization/batchnorm/add_1AddV2'batch_normalization/batchnorm/mul_1:z:0%batch_normalization/batchnorm/sub:z:0*
T0*+
_output_shapes
:          ѕ
leaky_re_lu/LeakyRelu	LeakyRelu'batch_normalization/batchnorm/add_1:z:0*+
_output_shapes
:          *
alpha%џЎЎ>i
conv1d_1/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§        ░
conv1d_1/Conv1D/ExpandDims
ExpandDims#leaky_re_lu/LeakyRelu:activations:0'conv1d_1/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:          ц
+conv1d_1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: @*
dtype0b
 conv1d_1/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ╗
conv1d_1/Conv1D/ExpandDims_1
ExpandDims3conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp:value:0)conv1d_1/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: @К
conv1d_1/Conv1DConv2D#conv1d_1/Conv1D/ExpandDims:output:0%conv1d_1/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
њ
conv1d_1/Conv1D/SqueezeSqueezeconv1d_1/Conv1D:output:0*
T0*+
_output_shapes
:         @*
squeeze_dims

§        ё
conv1d_1/BiasAdd/ReadVariableOpReadVariableOp(conv1d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0ю
conv1d_1/BiasAddBiasAdd conv1d_1/Conv1D/Squeeze:output:0'conv1d_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         @б
.batch_normalization_1/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_1_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0j
%batch_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:╣
#batch_normalization_1/batchnorm/addAddV26batch_normalization_1/batchnorm/ReadVariableOp:value:0.batch_normalization_1/batchnorm/add/y:output:0*
T0*
_output_shapes
:@|
%batch_normalization_1/batchnorm/RsqrtRsqrt'batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes
:@ф
2batch_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0Х
#batch_normalization_1/batchnorm/mulMul)batch_normalization_1/batchnorm/Rsqrt:y:0:batch_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@д
%batch_normalization_1/batchnorm/mul_1Mulconv1d_1/BiasAdd:output:0'batch_normalization_1/batchnorm/mul:z:0*
T0*+
_output_shapes
:         @д
0batch_normalization_1/batchnorm/ReadVariableOp_1ReadVariableOp9batch_normalization_1_batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype0┤
%batch_normalization_1/batchnorm/mul_2Mul8batch_normalization_1/batchnorm/ReadVariableOp_1:value:0'batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes
:@д
0batch_normalization_1/batchnorm/ReadVariableOp_2ReadVariableOp9batch_normalization_1_batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype0┤
#batch_normalization_1/batchnorm/subSub8batch_normalization_1/batchnorm/ReadVariableOp_2:value:0)batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@И
%batch_normalization_1/batchnorm/add_1AddV2)batch_normalization_1/batchnorm/mul_1:z:0'batch_normalization_1/batchnorm/sub:z:0*
T0*+
_output_shapes
:         @ї
leaky_re_lu_1/LeakyRelu	LeakyRelu)batch_normalization_1/batchnorm/add_1:z:0*+
_output_shapes
:         @*
alpha%џЎЎ>i
conv1d_2/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§        ▓
conv1d_2/Conv1D/ExpandDims
ExpandDims%leaky_re_lu_1/LeakyRelu:activations:0'conv1d_2/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:         @Ц
+conv1d_2/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_2_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:@ђ*
dtype0b
 conv1d_2/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ╝
conv1d_2/Conv1D/ExpandDims_1
ExpandDims3conv1d_2/Conv1D/ExpandDims_1/ReadVariableOp:value:0)conv1d_2/Conv1D/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:@ђ╚
conv1d_2/Conv1DConv2D#conv1d_2/Conv1D/ExpandDims:output:0%conv1d_2/Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:         ђ*
paddingSAME*
strides
Њ
conv1d_2/Conv1D/SqueezeSqueezeconv1d_2/Conv1D:output:0*
T0*,
_output_shapes
:         ђ*
squeeze_dims

§        Ё
conv1d_2/BiasAdd/ReadVariableOpReadVariableOp(conv1d_2_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0Ю
conv1d_2/BiasAddBiasAdd conv1d_2/Conv1D/Squeeze:output:0'conv1d_2/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:         ђБ
.batch_normalization_2/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_2_batchnorm_readvariableop_resource*
_output_shapes	
:ђ*
dtype0j
%batch_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:║
#batch_normalization_2/batchnorm/addAddV26batch_normalization_2/batchnorm/ReadVariableOp:value:0.batch_normalization_2/batchnorm/add/y:output:0*
T0*
_output_shapes	
:ђ}
%batch_normalization_2/batchnorm/RsqrtRsqrt'batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes	
:ђФ
2batch_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes	
:ђ*
dtype0и
#batch_normalization_2/batchnorm/mulMul)batch_normalization_2/batchnorm/Rsqrt:y:0:batch_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:ђД
%batch_normalization_2/batchnorm/mul_1Mulconv1d_2/BiasAdd:output:0'batch_normalization_2/batchnorm/mul:z:0*
T0*,
_output_shapes
:         ђД
0batch_normalization_2/batchnorm/ReadVariableOp_1ReadVariableOp9batch_normalization_2_batchnorm_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0х
%batch_normalization_2/batchnorm/mul_2Mul8batch_normalization_2/batchnorm/ReadVariableOp_1:value:0'batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes	
:ђД
0batch_normalization_2/batchnorm/ReadVariableOp_2ReadVariableOp9batch_normalization_2_batchnorm_readvariableop_2_resource*
_output_shapes	
:ђ*
dtype0х
#batch_normalization_2/batchnorm/subSub8batch_normalization_2/batchnorm/ReadVariableOp_2:value:0)batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:ђ╣
%batch_normalization_2/batchnorm/add_1AddV2)batch_normalization_2/batchnorm/mul_1:z:0'batch_normalization_2/batchnorm/sub:z:0*
T0*,
_output_shapes
:         ђЇ
leaky_re_lu_2/LeakyRelu	LeakyRelu)batch_normalization_2/batchnorm/add_1:z:0*,
_output_shapes
:         ђ*
alpha%џЎЎ>^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ї
flatten/ReshapeReshape%leaky_re_lu_2/LeakyRelu:activations:0flatten/Const:output:0*
T0*(
_output_shapes
:         ђІ
 z_log_mean/MatMul/ReadVariableOpReadVariableOp)z_log_mean_matmul_readvariableop_resource*
_output_shapes
:	ђ*
dtype0Љ
z_log_mean/MatMulMatMulflatten/Reshape:output:0(z_log_mean/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         ѕ
!z_log_mean/BiasAdd/ReadVariableOpReadVariableOp*z_log_mean_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ќ
z_log_mean/BiasAddBiasAddz_log_mean/MatMul:product:0)z_log_mean/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         Ѕ
z_log_var/MatMul/ReadVariableOpReadVariableOp(z_log_var_matmul_readvariableop_resource*
_output_shapes
:	ђ*
dtype0Ј
z_log_var/MatMulMatMulflatten/Reshape:output:0'z_log_var/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         є
 z_log_var/BiasAdd/ReadVariableOpReadVariableOp)z_log_var_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0ћ
z_log_var/BiasAddBiasAddz_log_var/MatMul:product:0(z_log_var/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         ▀
 sampling/StatefulPartitionedCallStatefulPartitionedCallz_log_mean/BiasAdd:output:0z_log_var/BiasAdd:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *2
f-R+
)__inference_restored_function_body_118440j
IdentityIdentityz_log_mean/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         k

Identity_1Identityz_log_var/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         z

Identity_2Identity)sampling/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         й
NoOpNoOp-^batch_normalization/batchnorm/ReadVariableOp/^batch_normalization/batchnorm/ReadVariableOp_1/^batch_normalization/batchnorm/ReadVariableOp_21^batch_normalization/batchnorm/mul/ReadVariableOp/^batch_normalization_1/batchnorm/ReadVariableOp1^batch_normalization_1/batchnorm/ReadVariableOp_11^batch_normalization_1/batchnorm/ReadVariableOp_23^batch_normalization_1/batchnorm/mul/ReadVariableOp/^batch_normalization_2/batchnorm/ReadVariableOp1^batch_normalization_2/batchnorm/ReadVariableOp_11^batch_normalization_2/batchnorm/ReadVariableOp_23^batch_normalization_2/batchnorm/mul/ReadVariableOp^conv1d/BiasAdd/ReadVariableOp*^conv1d/Conv1D/ExpandDims_1/ReadVariableOp ^conv1d_1/BiasAdd/ReadVariableOp,^conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp ^conv1d_2/BiasAdd/ReadVariableOp,^conv1d_2/Conv1D/ExpandDims_1/ReadVariableOp!^sampling/StatefulPartitionedCall"^z_log_mean/BiasAdd/ReadVariableOp!^z_log_mean/MatMul/ReadVariableOp!^z_log_var/BiasAdd/ReadVariableOp ^z_log_var/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:         : : : : : : : : : : : : : : : : : : : : : : 2\
,batch_normalization/batchnorm/ReadVariableOp,batch_normalization/batchnorm/ReadVariableOp2`
.batch_normalization/batchnorm/ReadVariableOp_1.batch_normalization/batchnorm/ReadVariableOp_12`
.batch_normalization/batchnorm/ReadVariableOp_2.batch_normalization/batchnorm/ReadVariableOp_22d
0batch_normalization/batchnorm/mul/ReadVariableOp0batch_normalization/batchnorm/mul/ReadVariableOp2`
.batch_normalization_1/batchnorm/ReadVariableOp.batch_normalization_1/batchnorm/ReadVariableOp2d
0batch_normalization_1/batchnorm/ReadVariableOp_10batch_normalization_1/batchnorm/ReadVariableOp_12d
0batch_normalization_1/batchnorm/ReadVariableOp_20batch_normalization_1/batchnorm/ReadVariableOp_22h
2batch_normalization_1/batchnorm/mul/ReadVariableOp2batch_normalization_1/batchnorm/mul/ReadVariableOp2`
.batch_normalization_2/batchnorm/ReadVariableOp.batch_normalization_2/batchnorm/ReadVariableOp2d
0batch_normalization_2/batchnorm/ReadVariableOp_10batch_normalization_2/batchnorm/ReadVariableOp_12d
0batch_normalization_2/batchnorm/ReadVariableOp_20batch_normalization_2/batchnorm/ReadVariableOp_22h
2batch_normalization_2/batchnorm/mul/ReadVariableOp2batch_normalization_2/batchnorm/mul/ReadVariableOp2>
conv1d/BiasAdd/ReadVariableOpconv1d/BiasAdd/ReadVariableOp2V
)conv1d/Conv1D/ExpandDims_1/ReadVariableOp)conv1d/Conv1D/ExpandDims_1/ReadVariableOp2B
conv1d_1/BiasAdd/ReadVariableOpconv1d_1/BiasAdd/ReadVariableOp2Z
+conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp+conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp2B
conv1d_2/BiasAdd/ReadVariableOpconv1d_2/BiasAdd/ReadVariableOp2Z
+conv1d_2/Conv1D/ExpandDims_1/ReadVariableOp+conv1d_2/Conv1D/ExpandDims_1/ReadVariableOp2D
 sampling/StatefulPartitionedCall sampling/StatefulPartitionedCall2F
!z_log_mean/BiasAdd/ReadVariableOp!z_log_mean/BiasAdd/ReadVariableOp2D
 z_log_mean/MatMul/ReadVariableOp z_log_mean/MatMul/ReadVariableOp2D
 z_log_var/BiasAdd/ReadVariableOp z_log_var/BiasAdd/ReadVariableOp2B
z_log_var/MatMul/ReadVariableOpz_log_var/MatMul/ReadVariableOp:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
Д
D
(__inference_flatten_layer_call_fn_120086

inputs
identity»
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_118815a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:         ђ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ђ:T P
,
_output_shapes
:         ђ
 
_user_specified_nameinputs
ы
c
G__inference_leaky_re_lu_layer_call_and_return_conditional_losses_119853

inputs
identity[
	LeakyRelu	LeakyReluinputs*+
_output_shapes
:          *
alpha%џЎЎ>c
IdentityIdentityLeakyRelu:activations:0*
T0*+
_output_shapes
:          "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:          :S O
+
_output_shapes
:          
 
_user_specified_nameinputs
Ј
░
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_119923

inputs/
!batchnorm_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@1
#batchnorm_readvariableop_1_resource:@1
#batchnorm_readvariableop_2_resource:@
identityѕбbatchnorm/ReadVariableOpбbatchnorm/ReadVariableOp_1бbatchnorm/ReadVariableOp_2бbatchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@p
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :                  @z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:@z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :                  @o
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :                  @║
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:                  @: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :                  @
 
_user_specified_nameinputs
ђ[
╠
"__inference__traced_restore_120297
file_prefix4
assignvariableop_conv1d_kernel: ,
assignvariableop_1_conv1d_bias: :
,assignvariableop_2_batch_normalization_gamma: 9
+assignvariableop_3_batch_normalization_beta: @
2assignvariableop_4_batch_normalization_moving_mean: D
6assignvariableop_5_batch_normalization_moving_variance: 8
"assignvariableop_6_conv1d_1_kernel: @.
 assignvariableop_7_conv1d_1_bias:@<
.assignvariableop_8_batch_normalization_1_gamma:@;
-assignvariableop_9_batch_normalization_1_beta:@C
5assignvariableop_10_batch_normalization_1_moving_mean:@G
9assignvariableop_11_batch_normalization_1_moving_variance:@:
#assignvariableop_12_conv1d_2_kernel:@ђ0
!assignvariableop_13_conv1d_2_bias:	ђ>
/assignvariableop_14_batch_normalization_2_gamma:	ђ=
.assignvariableop_15_batch_normalization_2_beta:	ђD
5assignvariableop_16_batch_normalization_2_moving_mean:	ђH
9assignvariableop_17_batch_normalization_2_moving_variance:	ђ8
%assignvariableop_18_z_log_mean_kernel:	ђ1
#assignvariableop_19_z_log_mean_bias:7
$assignvariableop_20_z_log_var_kernel:	ђ0
"assignvariableop_21_z_log_var_bias:
identity_23ѕбAssignVariableOpбAssignVariableOp_1бAssignVariableOp_10бAssignVariableOp_11бAssignVariableOp_12бAssignVariableOp_13бAssignVariableOp_14бAssignVariableOp_15бAssignVariableOp_16бAssignVariableOp_17бAssignVariableOp_18бAssignVariableOp_19бAssignVariableOp_2бAssignVariableOp_20бAssignVariableOp_21бAssignVariableOp_3бAssignVariableOp_4бAssignVariableOp_5бAssignVariableOp_6бAssignVariableOp_7бAssignVariableOp_8бAssignVariableOp_9З

RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*џ

valueљ
BЇ
B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHъ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*A
value8B6B B B B B B B B B B B B B B B B B B B B B B B Љ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*p
_output_shapes^
\:::::::::::::::::::::::*%
dtypes
2[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:Ѕ
AssignVariableOpAssignVariableOpassignvariableop_conv1d_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:Ї
AssignVariableOp_1AssignVariableOpassignvariableop_1_conv1d_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:Џ
AssignVariableOp_2AssignVariableOp,assignvariableop_2_batch_normalization_gammaIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:џ
AssignVariableOp_3AssignVariableOp+assignvariableop_3_batch_normalization_betaIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:А
AssignVariableOp_4AssignVariableOp2assignvariableop_4_batch_normalization_moving_meanIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:Ц
AssignVariableOp_5AssignVariableOp6assignvariableop_5_batch_normalization_moving_varianceIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:Љ
AssignVariableOp_6AssignVariableOp"assignvariableop_6_conv1d_1_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:Ј
AssignVariableOp_7AssignVariableOp assignvariableop_7_conv1d_1_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:Ю
AssignVariableOp_8AssignVariableOp.assignvariableop_8_batch_normalization_1_gammaIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:ю
AssignVariableOp_9AssignVariableOp-assignvariableop_9_batch_normalization_1_betaIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:д
AssignVariableOp_10AssignVariableOp5assignvariableop_10_batch_normalization_1_moving_meanIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:ф
AssignVariableOp_11AssignVariableOp9assignvariableop_11_batch_normalization_1_moving_varianceIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:ћ
AssignVariableOp_12AssignVariableOp#assignvariableop_12_conv1d_2_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:њ
AssignVariableOp_13AssignVariableOp!assignvariableop_13_conv1d_2_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:а
AssignVariableOp_14AssignVariableOp/assignvariableop_14_batch_normalization_2_gammaIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:Ъ
AssignVariableOp_15AssignVariableOp.assignvariableop_15_batch_normalization_2_betaIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:д
AssignVariableOp_16AssignVariableOp5assignvariableop_16_batch_normalization_2_moving_meanIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:ф
AssignVariableOp_17AssignVariableOp9assignvariableop_17_batch_normalization_2_moving_varianceIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:ќ
AssignVariableOp_18AssignVariableOp%assignvariableop_18_z_log_mean_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:ћ
AssignVariableOp_19AssignVariableOp#assignvariableop_19_z_log_mean_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:Ћ
AssignVariableOp_20AssignVariableOp$assignvariableop_20_z_log_var_kernelIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:Њ
AssignVariableOp_21AssignVariableOp"assignvariableop_21_z_log_var_biasIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 │
Identity_22Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_23IdentityIdentity_22:output:0^NoOp_1*
T0*
_output_shapes
: а
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_23Identity_23:output:0*A
_input_shapes0
.: : : : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_21AssignVariableOp_212(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
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
ы
c
G__inference_leaky_re_lu_layer_call_and_return_conditional_losses_118733

inputs
identity[
	LeakyRelu	LeakyReluinputs*+
_output_shapes
:          *
alpha%џЎЎ>c
IdentityIdentityLeakyRelu:activations:0*
T0*+
_output_shapes
:          "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:          :S O
+
_output_shapes
:          
 
_user_specified_nameinputs
═	
Э
F__inference_z_log_mean_layer_call_and_return_conditional_losses_118827

inputs1
matmul_readvariableop_resource:	ђ-
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	ђ*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ђ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
єC
╦

C__inference_encoder_layer_call_and_return_conditional_losses_119096

inputs#
conv1d_119036: 
conv1d_119038: (
batch_normalization_119041: (
batch_normalization_119043: (
batch_normalization_119045: (
batch_normalization_119047: %
conv1d_1_119051: @
conv1d_1_119053:@*
batch_normalization_1_119056:@*
batch_normalization_1_119058:@*
batch_normalization_1_119060:@*
batch_normalization_1_119062:@&
conv1d_2_119066:@ђ
conv1d_2_119068:	ђ+
batch_normalization_2_119071:	ђ+
batch_normalization_2_119073:	ђ+
batch_normalization_2_119075:	ђ+
batch_normalization_2_119077:	ђ$
z_log_mean_119082:	ђ
z_log_mean_119084:#
z_log_var_119087:	ђ
z_log_var_119089:
identity

identity_1

identity_2ѕб+batch_normalization/StatefulPartitionedCallб-batch_normalization_1/StatefulPartitionedCallб-batch_normalization_2/StatefulPartitionedCallбconv1d/StatefulPartitionedCallб conv1d_1/StatefulPartitionedCallб conv1d_2/StatefulPartitionedCallб sampling/StatefulPartitionedCallб"z_log_mean/StatefulPartitionedCallб!z_log_var/StatefulPartitionedCallВ
conv1d/StatefulPartitionedCallStatefulPartitionedCallinputsconv1d_119036conv1d_119038*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:          *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *K
fFRD
B__inference_conv1d_layer_call_and_return_conditional_losses_118713ч
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall'conv1d/StatefulPartitionedCall:output:0batch_normalization_119041batch_normalization_119043batch_normalization_119045batch_normalization_119047*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:          *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_118516­
leaky_re_lu/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:          * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *P
fKRI
G__inference_leaky_re_lu_layer_call_and_return_conditional_losses_118733њ
 conv1d_1/StatefulPartitionedCallStatefulPartitionedCall$leaky_re_lu/PartitionedCall:output:0conv1d_1_119051conv1d_1_119053*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_conv1d_1_layer_call_and_return_conditional_losses_118750Ѕ
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall)conv1d_1/StatefulPartitionedCall:output:0batch_normalization_1_119056batch_normalization_1_119058batch_normalization_1_119060batch_normalization_1_119062*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *Z
fURS
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_118598Ш
leaky_re_lu_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *R
fMRK
I__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_118770Ћ
 conv1d_2/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_1/PartitionedCall:output:0conv1d_2_119066conv1d_2_119068*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ђ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_conv1d_2_layer_call_and_return_conditional_losses_118787і
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall)conv1d_2/StatefulPartitionedCall:output:0batch_normalization_2_119071batch_normalization_2_119073batch_normalization_2_119075batch_normalization_2_119077*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ђ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *Z
fURS
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_118680э
leaky_re_lu_2/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *R
fMRK
I__inference_leaky_re_lu_2_layer_call_and_return_conditional_losses_118807О
flatten/PartitionedCallPartitionedCall&leaky_re_lu_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_118815њ
"z_log_mean/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0z_log_mean_119082z_log_mean_119084*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *O
fJRH
F__inference_z_log_mean_layer_call_and_return_conditional_losses_118827ј
!z_log_var/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0z_log_var_119087z_log_var_119089*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *N
fIRG
E__inference_z_log_var_layer_call_and_return_conditional_losses_118843 
 sampling/StatefulPartitionedCallStatefulPartitionedCall+z_log_mean/StatefulPartitionedCall:output:0*z_log_var/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *2
f-R+
)__inference_restored_function_body_118440z
IdentityIdentity+z_log_mean/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         {

Identity_1Identity*z_log_var/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         z

Identity_2Identity)sampling/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         Д
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall^conv1d/StatefulPartitionedCall!^conv1d_1/StatefulPartitionedCall!^conv1d_2/StatefulPartitionedCall!^sampling/StatefulPartitionedCall#^z_log_mean/StatefulPartitionedCall"^z_log_var/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:         : : : : : : : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2@
conv1d/StatefulPartitionedCallconv1d/StatefulPartitionedCall2D
 conv1d_1/StatefulPartitionedCall conv1d_1/StatefulPartitionedCall2D
 conv1d_2/StatefulPartitionedCall conv1d_2/StatefulPartitionedCall2D
 sampling/StatefulPartitionedCall sampling/StatefulPartitionedCall2H
"z_log_mean/StatefulPartitionedCall"z_log_mean/StatefulPartitionedCall2F
!z_log_var/StatefulPartitionedCall!z_log_var/StatefulPartitionedCall:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
╔
Ў
+__inference_z_log_mean_layer_call_fn_120101

inputs
unknown:	ђ
	unknown_0:
identityѕбStatefulPartitionedCall█
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *O
fJRH
F__inference_z_log_mean_layer_call_and_return_conditional_losses_118827o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ђ: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
Ўц
а
!__inference__wrapped_model_118445
input_1P
:encoder_conv1d_conv1d_expanddims_1_readvariableop_resource: <
.encoder_conv1d_biasadd_readvariableop_resource: K
=encoder_batch_normalization_batchnorm_readvariableop_resource: O
Aencoder_batch_normalization_batchnorm_mul_readvariableop_resource: M
?encoder_batch_normalization_batchnorm_readvariableop_1_resource: M
?encoder_batch_normalization_batchnorm_readvariableop_2_resource: R
<encoder_conv1d_1_conv1d_expanddims_1_readvariableop_resource: @>
0encoder_conv1d_1_biasadd_readvariableop_resource:@M
?encoder_batch_normalization_1_batchnorm_readvariableop_resource:@Q
Cencoder_batch_normalization_1_batchnorm_mul_readvariableop_resource:@O
Aencoder_batch_normalization_1_batchnorm_readvariableop_1_resource:@O
Aencoder_batch_normalization_1_batchnorm_readvariableop_2_resource:@S
<encoder_conv1d_2_conv1d_expanddims_1_readvariableop_resource:@ђ?
0encoder_conv1d_2_biasadd_readvariableop_resource:	ђN
?encoder_batch_normalization_2_batchnorm_readvariableop_resource:	ђR
Cencoder_batch_normalization_2_batchnorm_mul_readvariableop_resource:	ђP
Aencoder_batch_normalization_2_batchnorm_readvariableop_1_resource:	ђP
Aencoder_batch_normalization_2_batchnorm_readvariableop_2_resource:	ђD
1encoder_z_log_mean_matmul_readvariableop_resource:	ђ@
2encoder_z_log_mean_biasadd_readvariableop_resource:C
0encoder_z_log_var_matmul_readvariableop_resource:	ђ?
1encoder_z_log_var_biasadd_readvariableop_resource:
identity

identity_1

identity_2ѕб4encoder/batch_normalization/batchnorm/ReadVariableOpб6encoder/batch_normalization/batchnorm/ReadVariableOp_1б6encoder/batch_normalization/batchnorm/ReadVariableOp_2б8encoder/batch_normalization/batchnorm/mul/ReadVariableOpб6encoder/batch_normalization_1/batchnorm/ReadVariableOpб8encoder/batch_normalization_1/batchnorm/ReadVariableOp_1б8encoder/batch_normalization_1/batchnorm/ReadVariableOp_2б:encoder/batch_normalization_1/batchnorm/mul/ReadVariableOpб6encoder/batch_normalization_2/batchnorm/ReadVariableOpб8encoder/batch_normalization_2/batchnorm/ReadVariableOp_1б8encoder/batch_normalization_2/batchnorm/ReadVariableOp_2б:encoder/batch_normalization_2/batchnorm/mul/ReadVariableOpб%encoder/conv1d/BiasAdd/ReadVariableOpб1encoder/conv1d/Conv1D/ExpandDims_1/ReadVariableOpб'encoder/conv1d_1/BiasAdd/ReadVariableOpб3encoder/conv1d_1/Conv1D/ExpandDims_1/ReadVariableOpб'encoder/conv1d_2/BiasAdd/ReadVariableOpб3encoder/conv1d_2/Conv1D/ExpandDims_1/ReadVariableOpб(encoder/sampling/StatefulPartitionedCallб)encoder/z_log_mean/BiasAdd/ReadVariableOpб(encoder/z_log_mean/MatMul/ReadVariableOpб(encoder/z_log_var/BiasAdd/ReadVariableOpб'encoder/z_log_var/MatMul/ReadVariableOpo
$encoder/conv1d/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§        а
 encoder/conv1d/Conv1D/ExpandDims
ExpandDimsinput_1-encoder/conv1d/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:         ░
1encoder/conv1d/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp:encoder_conv1d_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype0h
&encoder/conv1d/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ═
"encoder/conv1d/Conv1D/ExpandDims_1
ExpandDims9encoder/conv1d/Conv1D/ExpandDims_1/ReadVariableOp:value:0/encoder/conv1d/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: ┘
encoder/conv1d/Conv1DConv2D)encoder/conv1d/Conv1D/ExpandDims:output:0+encoder/conv1d/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:          *
paddingSAME*
strides
ъ
encoder/conv1d/Conv1D/SqueezeSqueezeencoder/conv1d/Conv1D:output:0*
T0*+
_output_shapes
:          *
squeeze_dims

§        љ
%encoder/conv1d/BiasAdd/ReadVariableOpReadVariableOp.encoder_conv1d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0«
encoder/conv1d/BiasAddBiasAdd&encoder/conv1d/Conv1D/Squeeze:output:0-encoder/conv1d/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:          «
4encoder/batch_normalization/batchnorm/ReadVariableOpReadVariableOp=encoder_batch_normalization_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype0p
+encoder/batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:╦
)encoder/batch_normalization/batchnorm/addAddV2<encoder/batch_normalization/batchnorm/ReadVariableOp:value:04encoder/batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes
: ѕ
+encoder/batch_normalization/batchnorm/RsqrtRsqrt-encoder/batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes
: Х
8encoder/batch_normalization/batchnorm/mul/ReadVariableOpReadVariableOpAencoder_batch_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype0╚
)encoder/batch_normalization/batchnorm/mulMul/encoder/batch_normalization/batchnorm/Rsqrt:y:0@encoder/batch_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: И
+encoder/batch_normalization/batchnorm/mul_1Mulencoder/conv1d/BiasAdd:output:0-encoder/batch_normalization/batchnorm/mul:z:0*
T0*+
_output_shapes
:          ▓
6encoder/batch_normalization/batchnorm/ReadVariableOp_1ReadVariableOp?encoder_batch_normalization_batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype0к
+encoder/batch_normalization/batchnorm/mul_2Mul>encoder/batch_normalization/batchnorm/ReadVariableOp_1:value:0-encoder/batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes
: ▓
6encoder/batch_normalization/batchnorm/ReadVariableOp_2ReadVariableOp?encoder_batch_normalization_batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype0к
)encoder/batch_normalization/batchnorm/subSub>encoder/batch_normalization/batchnorm/ReadVariableOp_2:value:0/encoder/batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes
: ╩
+encoder/batch_normalization/batchnorm/add_1AddV2/encoder/batch_normalization/batchnorm/mul_1:z:0-encoder/batch_normalization/batchnorm/sub:z:0*
T0*+
_output_shapes
:          ў
encoder/leaky_re_lu/LeakyRelu	LeakyRelu/encoder/batch_normalization/batchnorm/add_1:z:0*+
_output_shapes
:          *
alpha%џЎЎ>q
&encoder/conv1d_1/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§        ╚
"encoder/conv1d_1/Conv1D/ExpandDims
ExpandDims+encoder/leaky_re_lu/LeakyRelu:activations:0/encoder/conv1d_1/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:          ┤
3encoder/conv1d_1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp<encoder_conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: @*
dtype0j
(encoder/conv1d_1/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : М
$encoder/conv1d_1/Conv1D/ExpandDims_1
ExpandDims;encoder/conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp:value:01encoder/conv1d_1/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: @▀
encoder/conv1d_1/Conv1DConv2D+encoder/conv1d_1/Conv1D/ExpandDims:output:0-encoder/conv1d_1/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
б
encoder/conv1d_1/Conv1D/SqueezeSqueeze encoder/conv1d_1/Conv1D:output:0*
T0*+
_output_shapes
:         @*
squeeze_dims

§        ћ
'encoder/conv1d_1/BiasAdd/ReadVariableOpReadVariableOp0encoder_conv1d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0┤
encoder/conv1d_1/BiasAddBiasAdd(encoder/conv1d_1/Conv1D/Squeeze:output:0/encoder/conv1d_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         @▓
6encoder/batch_normalization_1/batchnorm/ReadVariableOpReadVariableOp?encoder_batch_normalization_1_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0r
-encoder/batch_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:Л
+encoder/batch_normalization_1/batchnorm/addAddV2>encoder/batch_normalization_1/batchnorm/ReadVariableOp:value:06encoder/batch_normalization_1/batchnorm/add/y:output:0*
T0*
_output_shapes
:@ї
-encoder/batch_normalization_1/batchnorm/RsqrtRsqrt/encoder/batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes
:@║
:encoder/batch_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOpCencoder_batch_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0╬
+encoder/batch_normalization_1/batchnorm/mulMul1encoder/batch_normalization_1/batchnorm/Rsqrt:y:0Bencoder/batch_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@Й
-encoder/batch_normalization_1/batchnorm/mul_1Mul!encoder/conv1d_1/BiasAdd:output:0/encoder/batch_normalization_1/batchnorm/mul:z:0*
T0*+
_output_shapes
:         @Х
8encoder/batch_normalization_1/batchnorm/ReadVariableOp_1ReadVariableOpAencoder_batch_normalization_1_batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype0╠
-encoder/batch_normalization_1/batchnorm/mul_2Mul@encoder/batch_normalization_1/batchnorm/ReadVariableOp_1:value:0/encoder/batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes
:@Х
8encoder/batch_normalization_1/batchnorm/ReadVariableOp_2ReadVariableOpAencoder_batch_normalization_1_batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype0╠
+encoder/batch_normalization_1/batchnorm/subSub@encoder/batch_normalization_1/batchnorm/ReadVariableOp_2:value:01encoder/batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@л
-encoder/batch_normalization_1/batchnorm/add_1AddV21encoder/batch_normalization_1/batchnorm/mul_1:z:0/encoder/batch_normalization_1/batchnorm/sub:z:0*
T0*+
_output_shapes
:         @ю
encoder/leaky_re_lu_1/LeakyRelu	LeakyRelu1encoder/batch_normalization_1/batchnorm/add_1:z:0*+
_output_shapes
:         @*
alpha%џЎЎ>q
&encoder/conv1d_2/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§        ╩
"encoder/conv1d_2/Conv1D/ExpandDims
ExpandDims-encoder/leaky_re_lu_1/LeakyRelu:activations:0/encoder/conv1d_2/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:         @х
3encoder/conv1d_2/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp<encoder_conv1d_2_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:@ђ*
dtype0j
(encoder/conv1d_2/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : н
$encoder/conv1d_2/Conv1D/ExpandDims_1
ExpandDims;encoder/conv1d_2/Conv1D/ExpandDims_1/ReadVariableOp:value:01encoder/conv1d_2/Conv1D/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:@ђЯ
encoder/conv1d_2/Conv1DConv2D+encoder/conv1d_2/Conv1D/ExpandDims:output:0-encoder/conv1d_2/Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:         ђ*
paddingSAME*
strides
Б
encoder/conv1d_2/Conv1D/SqueezeSqueeze encoder/conv1d_2/Conv1D:output:0*
T0*,
_output_shapes
:         ђ*
squeeze_dims

§        Ћ
'encoder/conv1d_2/BiasAdd/ReadVariableOpReadVariableOp0encoder_conv1d_2_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0х
encoder/conv1d_2/BiasAddBiasAdd(encoder/conv1d_2/Conv1D/Squeeze:output:0/encoder/conv1d_2/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:         ђ│
6encoder/batch_normalization_2/batchnorm/ReadVariableOpReadVariableOp?encoder_batch_normalization_2_batchnorm_readvariableop_resource*
_output_shapes	
:ђ*
dtype0r
-encoder/batch_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:м
+encoder/batch_normalization_2/batchnorm/addAddV2>encoder/batch_normalization_2/batchnorm/ReadVariableOp:value:06encoder/batch_normalization_2/batchnorm/add/y:output:0*
T0*
_output_shapes	
:ђЇ
-encoder/batch_normalization_2/batchnorm/RsqrtRsqrt/encoder/batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes	
:ђ╗
:encoder/batch_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOpCencoder_batch_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes	
:ђ*
dtype0¤
+encoder/batch_normalization_2/batchnorm/mulMul1encoder/batch_normalization_2/batchnorm/Rsqrt:y:0Bencoder/batch_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:ђ┐
-encoder/batch_normalization_2/batchnorm/mul_1Mul!encoder/conv1d_2/BiasAdd:output:0/encoder/batch_normalization_2/batchnorm/mul:z:0*
T0*,
_output_shapes
:         ђи
8encoder/batch_normalization_2/batchnorm/ReadVariableOp_1ReadVariableOpAencoder_batch_normalization_2_batchnorm_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0═
-encoder/batch_normalization_2/batchnorm/mul_2Mul@encoder/batch_normalization_2/batchnorm/ReadVariableOp_1:value:0/encoder/batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes	
:ђи
8encoder/batch_normalization_2/batchnorm/ReadVariableOp_2ReadVariableOpAencoder_batch_normalization_2_batchnorm_readvariableop_2_resource*
_output_shapes	
:ђ*
dtype0═
+encoder/batch_normalization_2/batchnorm/subSub@encoder/batch_normalization_2/batchnorm/ReadVariableOp_2:value:01encoder/batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:ђЛ
-encoder/batch_normalization_2/batchnorm/add_1AddV21encoder/batch_normalization_2/batchnorm/mul_1:z:0/encoder/batch_normalization_2/batchnorm/sub:z:0*
T0*,
_output_shapes
:         ђЮ
encoder/leaky_re_lu_2/LeakyRelu	LeakyRelu1encoder/batch_normalization_2/batchnorm/add_1:z:0*,
_output_shapes
:         ђ*
alpha%џЎЎ>f
encoder/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ц
encoder/flatten/ReshapeReshape-encoder/leaky_re_lu_2/LeakyRelu:activations:0encoder/flatten/Const:output:0*
T0*(
_output_shapes
:         ђЏ
(encoder/z_log_mean/MatMul/ReadVariableOpReadVariableOp1encoder_z_log_mean_matmul_readvariableop_resource*
_output_shapes
:	ђ*
dtype0Е
encoder/z_log_mean/MatMulMatMul encoder/flatten/Reshape:output:00encoder/z_log_mean/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         ў
)encoder/z_log_mean/BiasAdd/ReadVariableOpReadVariableOp2encoder_z_log_mean_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0»
encoder/z_log_mean/BiasAddBiasAdd#encoder/z_log_mean/MatMul:product:01encoder/z_log_mean/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         Ў
'encoder/z_log_var/MatMul/ReadVariableOpReadVariableOp0encoder_z_log_var_matmul_readvariableop_resource*
_output_shapes
:	ђ*
dtype0Д
encoder/z_log_var/MatMulMatMul encoder/flatten/Reshape:output:0/encoder/z_log_var/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         ќ
(encoder/z_log_var/BiasAdd/ReadVariableOpReadVariableOp1encoder_z_log_var_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0г
encoder/z_log_var/BiasAddBiasAdd"encoder/z_log_var/MatMul:product:00encoder/z_log_var/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         э
(encoder/sampling/StatefulPartitionedCallStatefulPartitionedCall#encoder/z_log_mean/BiasAdd:output:0"encoder/z_log_var/BiasAdd:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *2
f-R+
)__inference_restored_function_body_118440ђ
IdentityIdentity1encoder/sampling/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         t

Identity_1Identity#encoder/z_log_mean/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         s

Identity_2Identity"encoder/z_log_var/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         ш	
NoOpNoOp5^encoder/batch_normalization/batchnorm/ReadVariableOp7^encoder/batch_normalization/batchnorm/ReadVariableOp_17^encoder/batch_normalization/batchnorm/ReadVariableOp_29^encoder/batch_normalization/batchnorm/mul/ReadVariableOp7^encoder/batch_normalization_1/batchnorm/ReadVariableOp9^encoder/batch_normalization_1/batchnorm/ReadVariableOp_19^encoder/batch_normalization_1/batchnorm/ReadVariableOp_2;^encoder/batch_normalization_1/batchnorm/mul/ReadVariableOp7^encoder/batch_normalization_2/batchnorm/ReadVariableOp9^encoder/batch_normalization_2/batchnorm/ReadVariableOp_19^encoder/batch_normalization_2/batchnorm/ReadVariableOp_2;^encoder/batch_normalization_2/batchnorm/mul/ReadVariableOp&^encoder/conv1d/BiasAdd/ReadVariableOp2^encoder/conv1d/Conv1D/ExpandDims_1/ReadVariableOp(^encoder/conv1d_1/BiasAdd/ReadVariableOp4^encoder/conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp(^encoder/conv1d_2/BiasAdd/ReadVariableOp4^encoder/conv1d_2/Conv1D/ExpandDims_1/ReadVariableOp)^encoder/sampling/StatefulPartitionedCall*^encoder/z_log_mean/BiasAdd/ReadVariableOp)^encoder/z_log_mean/MatMul/ReadVariableOp)^encoder/z_log_var/BiasAdd/ReadVariableOp(^encoder/z_log_var/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:         : : : : : : : : : : : : : : : : : : : : : : 2l
4encoder/batch_normalization/batchnorm/ReadVariableOp4encoder/batch_normalization/batchnorm/ReadVariableOp2p
6encoder/batch_normalization/batchnorm/ReadVariableOp_16encoder/batch_normalization/batchnorm/ReadVariableOp_12p
6encoder/batch_normalization/batchnorm/ReadVariableOp_26encoder/batch_normalization/batchnorm/ReadVariableOp_22t
8encoder/batch_normalization/batchnorm/mul/ReadVariableOp8encoder/batch_normalization/batchnorm/mul/ReadVariableOp2p
6encoder/batch_normalization_1/batchnorm/ReadVariableOp6encoder/batch_normalization_1/batchnorm/ReadVariableOp2t
8encoder/batch_normalization_1/batchnorm/ReadVariableOp_18encoder/batch_normalization_1/batchnorm/ReadVariableOp_12t
8encoder/batch_normalization_1/batchnorm/ReadVariableOp_28encoder/batch_normalization_1/batchnorm/ReadVariableOp_22x
:encoder/batch_normalization_1/batchnorm/mul/ReadVariableOp:encoder/batch_normalization_1/batchnorm/mul/ReadVariableOp2p
6encoder/batch_normalization_2/batchnorm/ReadVariableOp6encoder/batch_normalization_2/batchnorm/ReadVariableOp2t
8encoder/batch_normalization_2/batchnorm/ReadVariableOp_18encoder/batch_normalization_2/batchnorm/ReadVariableOp_12t
8encoder/batch_normalization_2/batchnorm/ReadVariableOp_28encoder/batch_normalization_2/batchnorm/ReadVariableOp_22x
:encoder/batch_normalization_2/batchnorm/mul/ReadVariableOp:encoder/batch_normalization_2/batchnorm/mul/ReadVariableOp2N
%encoder/conv1d/BiasAdd/ReadVariableOp%encoder/conv1d/BiasAdd/ReadVariableOp2f
1encoder/conv1d/Conv1D/ExpandDims_1/ReadVariableOp1encoder/conv1d/Conv1D/ExpandDims_1/ReadVariableOp2R
'encoder/conv1d_1/BiasAdd/ReadVariableOp'encoder/conv1d_1/BiasAdd/ReadVariableOp2j
3encoder/conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp3encoder/conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp2R
'encoder/conv1d_2/BiasAdd/ReadVariableOp'encoder/conv1d_2/BiasAdd/ReadVariableOp2j
3encoder/conv1d_2/Conv1D/ExpandDims_1/ReadVariableOp3encoder/conv1d_2/Conv1D/ExpandDims_1/ReadVariableOp2T
(encoder/sampling/StatefulPartitionedCall(encoder/sampling/StatefulPartitionedCall2V
)encoder/z_log_mean/BiasAdd/ReadVariableOp)encoder/z_log_mean/BiasAdd/ReadVariableOp2T
(encoder/z_log_mean/MatMul/ReadVariableOp(encoder/z_log_mean/MatMul/ReadVariableOp2T
(encoder/z_log_var/BiasAdd/ReadVariableOp(encoder/z_log_var/BiasAdd/ReadVariableOp2R
'encoder/z_log_var/MatMul/ReadVariableOp'encoder/z_log_var/MatMul/ReadVariableOp:T P
+
_output_shapes
:         
!
_user_specified_name	input_1
ь
Љ
B__inference_conv1d_layer_call_and_return_conditional_losses_119763

inputsA
+conv1d_expanddims_1_readvariableop_resource: -
biasadd_readvariableop_resource: 
identityѕбBiasAdd/ReadVariableOpб"Conv1D/ExpandDims_1/ReadVariableOp`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§        Ђ
Conv1D/ExpandDims
ExpandDimsinputsConv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:         њ
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : а
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: г
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:          *
paddingSAME*
strides
ђ
Conv1D/SqueezeSqueezeConv1D:output:0*
T0*+
_output_shapes
:          *
squeeze_dims

§        r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0Ђ
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:          c
IdentityIdentityBiasAdd:output:0^NoOp*
T0*+
_output_shapes
:          ё
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
§%
Ж
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_119957

inputs5
'assignmovingavg_readvariableop_resource:@7
)assignmovingavg_1_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@/
!batchnorm_readvariableop_resource:@
identityѕбAssignMovingAvgбAssignMovingAvg/ReadVariableOpбAssignMovingAvg_1б AssignMovingAvg_1/ReadVariableOpбbatchnorm/ReadVariableOpбbatchnorm/mul/ReadVariableOpo
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       Ѓ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(h
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
:@ћ
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*4
_output_shapes"
 :                  @s
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       б
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(n
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 t
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<ѓ
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:@*
dtype0Ђ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:@x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:@г
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
О#<є
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:@*
dtype0Є
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:@~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:@┤
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@p
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :                  @h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:@v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :                  @o
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :                  @Ж
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:                  @: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :                  @
 
_user_specified_nameinputs
ЈC
╠

C__inference_encoder_layer_call_and_return_conditional_losses_119263
input_1#
conv1d_119203: 
conv1d_119205: (
batch_normalization_119208: (
batch_normalization_119210: (
batch_normalization_119212: (
batch_normalization_119214: %
conv1d_1_119218: @
conv1d_1_119220:@*
batch_normalization_1_119223:@*
batch_normalization_1_119225:@*
batch_normalization_1_119227:@*
batch_normalization_1_119229:@&
conv1d_2_119233:@ђ
conv1d_2_119235:	ђ+
batch_normalization_2_119238:	ђ+
batch_normalization_2_119240:	ђ+
batch_normalization_2_119242:	ђ+
batch_normalization_2_119244:	ђ$
z_log_mean_119249:	ђ
z_log_mean_119251:#
z_log_var_119254:	ђ
z_log_var_119256:
identity

identity_1

identity_2ѕб+batch_normalization/StatefulPartitionedCallб-batch_normalization_1/StatefulPartitionedCallб-batch_normalization_2/StatefulPartitionedCallбconv1d/StatefulPartitionedCallб conv1d_1/StatefulPartitionedCallб conv1d_2/StatefulPartitionedCallб sampling/StatefulPartitionedCallб"z_log_mean/StatefulPartitionedCallб!z_log_var/StatefulPartitionedCallь
conv1d/StatefulPartitionedCallStatefulPartitionedCallinput_1conv1d_119203conv1d_119205*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:          *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *K
fFRD
B__inference_conv1d_layer_call_and_return_conditional_losses_118713§
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall'conv1d/StatefulPartitionedCall:output:0batch_normalization_119208batch_normalization_119210batch_normalization_119212batch_normalization_119214*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:          *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_118469­
leaky_re_lu/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:          * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *P
fKRI
G__inference_leaky_re_lu_layer_call_and_return_conditional_losses_118733њ
 conv1d_1/StatefulPartitionedCallStatefulPartitionedCall$leaky_re_lu/PartitionedCall:output:0conv1d_1_119218conv1d_1_119220*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_conv1d_1_layer_call_and_return_conditional_losses_118750І
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall)conv1d_1/StatefulPartitionedCall:output:0batch_normalization_1_119223batch_normalization_1_119225batch_normalization_1_119227batch_normalization_1_119229*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         @*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *Z
fURS
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_118551Ш
leaky_re_lu_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *R
fMRK
I__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_118770Ћ
 conv1d_2/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_1/PartitionedCall:output:0conv1d_2_119233conv1d_2_119235*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ђ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_conv1d_2_layer_call_and_return_conditional_losses_118787ї
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall)conv1d_2/StatefulPartitionedCall:output:0batch_normalization_2_119238batch_normalization_2_119240batch_normalization_2_119242batch_normalization_2_119244*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ђ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *Z
fURS
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_118633э
leaky_re_lu_2/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *R
fMRK
I__inference_leaky_re_lu_2_layer_call_and_return_conditional_losses_118807О
flatten/PartitionedCallPartitionedCall&leaky_re_lu_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_118815њ
"z_log_mean/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0z_log_mean_119249z_log_mean_119251*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *O
fJRH
F__inference_z_log_mean_layer_call_and_return_conditional_losses_118827ј
!z_log_var/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0z_log_var_119254z_log_var_119256*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *N
fIRG
E__inference_z_log_var_layer_call_and_return_conditional_losses_118843 
 sampling/StatefulPartitionedCallStatefulPartitionedCall+z_log_mean/StatefulPartitionedCall:output:0*z_log_var/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *2
f-R+
)__inference_restored_function_body_118440z
IdentityIdentity+z_log_mean/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         {

Identity_1Identity*z_log_var/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         z

Identity_2Identity)sampling/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         Д
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall^conv1d/StatefulPartitionedCall!^conv1d_1/StatefulPartitionedCall!^conv1d_2/StatefulPartitionedCall!^sampling/StatefulPartitionedCall#^z_log_mean/StatefulPartitionedCall"^z_log_var/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:         : : : : : : : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2@
conv1d/StatefulPartitionedCallconv1d/StatefulPartitionedCall2D
 conv1d_1/StatefulPartitionedCall conv1d_1/StatefulPartitionedCall2D
 conv1d_2/StatefulPartitionedCall conv1d_2/StatefulPartitionedCall2D
 sampling/StatefulPartitionedCall sampling/StatefulPartitionedCall2H
"z_log_mean/StatefulPartitionedCall"z_log_mean/StatefulPartitionedCall2F
!z_log_var/StatefulPartitionedCall!z_log_var/StatefulPartitionedCall:T P
+
_output_shapes
:         
!
_user_specified_name	input_1
ќ
п
(__inference_encoder_layer_call_fn_119379

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: 
	unknown_5: @
	unknown_6:@
	unknown_7:@
	unknown_8:@
	unknown_9:@

unknown_10:@!

unknown_11:@ђ

unknown_12:	ђ

unknown_13:	ђ

unknown_14:	ђ

unknown_15:	ђ

unknown_16:	ђ

unknown_17:	ђ

unknown_18:

unknown_19:	ђ

unknown_20:
identity

identity_1

identity_2ѕбStatefulPartitionedCallЈ
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
unknown_20*"
Tin
2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:         :         :         *8
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_encoder_layer_call_and_return_conditional_losses_118853o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:         q

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:         : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
о
џ
)__inference_conv1d_1_layer_call_fn_119862

inputs
unknown: @
	unknown_0:@
identityѕбStatefulPartitionedCallП
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_conv1d_1_layer_call_and_return_conditional_losses_118750s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         @`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:          : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:          
 
_user_specified_nameinputs
џ&
Ь
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_118680

inputs6
'assignmovingavg_readvariableop_resource:	ђ8
)assignmovingavg_1_readvariableop_resource:	ђ4
%batchnorm_mul_readvariableop_resource:	ђ0
!batchnorm_readvariableop_resource:	ђ
identityѕбAssignMovingAvgбAssignMovingAvg/ReadVariableOpбAssignMovingAvg_1б AssignMovingAvg_1/ReadVariableOpбbatchnorm/ReadVariableOpбbatchnorm/mul/ReadVariableOpo
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       ё
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*#
_output_shapes
:ђ*
	keep_dims(i
moments/StopGradientStopGradientmoments/mean:output:0*
T0*#
_output_shapes
:ђЋ
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*5
_output_shapes#
!:                  ђs
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       Б
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*#
_output_shapes
:ђ*
	keep_dims(o
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:ђ*
squeeze_dims
 u
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:ђ*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<Ѓ
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:ђ*
dtype0ѓ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:ђy
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:ђг
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
О#<Є
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:ђ*
dtype0ѕ
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:ђ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:ђ┤
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:r
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:ђQ
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:ђ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:ђ*
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:ђq
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*5
_output_shapes#
!:                  ђi
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:ђw
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:ђ*
dtype0q
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:ђђ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*5
_output_shapes#
!:                  ђp
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*5
_output_shapes#
!:                  ђЖ
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):                  ђ: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:] Y
5
_output_shapes#
!:                  ђ
 
_user_specified_nameinputs
Ў
┘
(__inference_encoder_layer_call_fn_118904
input_1
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: 
	unknown_5: @
	unknown_6:@
	unknown_7:@
	unknown_8:@
	unknown_9:@

unknown_10:@!

unknown_11:@ђ

unknown_12:	ђ

unknown_13:	ђ

unknown_14:	ђ

unknown_15:	ђ

unknown_16:	ђ

unknown_17:	ђ

unknown_18:

unknown_19:	ђ

unknown_20:
identity

identity_1

identity_2ѕбStatefulPartitionedCallљ
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_20*"
Tin
2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:         :         :         *8
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_encoder_layer_call_and_return_conditional_losses_118853o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:         q

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:         : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:         
!
_user_specified_name	input_1
љ
п
(__inference_encoder_layer_call_fn_119432

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: 
	unknown_5: @
	unknown_6:@
	unknown_7:@
	unknown_8:@
	unknown_9:@

unknown_10:@!

unknown_11:@ђ

unknown_12:	ђ

unknown_13:	ђ

unknown_14:	ђ

unknown_15:	ђ

unknown_16:	ђ

unknown_17:	ђ

unknown_18:

unknown_19:	ђ

unknown_20:
identity

identity_1

identity_2ѕбStatefulPartitionedCallЅ
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
unknown_20*"
Tin
2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:         :         :         *2
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_encoder_layer_call_and_return_conditional_losses_119096o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:         q

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:         : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
Э
Ћ
D__inference_conv1d_2_layer_call_and_return_conditional_losses_118787

inputsB
+conv1d_expanddims_1_readvariableop_resource:@ђ.
biasadd_readvariableop_resource:	ђ
identityѕбBiasAdd/ReadVariableOpб"Conv1D/ExpandDims_1/ReadVariableOp`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§        Ђ
Conv1D/ExpandDims
ExpandDimsinputsConv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:         @Њ
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:@ђ*
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : А
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:@ђГ
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:         ђ*
paddingSAME*
strides
Ђ
Conv1D/SqueezeSqueezeConv1D:output:0*
T0*,
_output_shapes
:         ђ*
squeeze_dims

§        s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0ѓ
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:         ђd
IdentityIdentityBiasAdd:output:0^NoOp*
T0*,
_output_shapes
:         ђё
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         @: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:         @
 
_user_specified_nameinputs
э
e
I__inference_leaky_re_lu_2_layer_call_and_return_conditional_losses_118807

inputs
identity\
	LeakyRelu	LeakyReluinputs*,
_output_shapes
:         ђ*
alpha%џЎЎ>d
IdentityIdentityLeakyRelu:activations:0*
T0*,
_output_shapes
:         ђ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ђ:T P
,
_output_shapes
:         ђ
 
_user_specified_nameinputs
╗
J
.__inference_leaky_re_lu_2_layer_call_fn_120076

inputs
identity╣
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *R
fMRK
I__inference_leaky_re_lu_2_layer_call_and_return_conditional_losses_118807e
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:         ђ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ђ:T P
,
_output_shapes
:         ђ
 
_user_specified_nameinputs
Э
Ћ
D__inference_conv1d_2_layer_call_and_return_conditional_losses_119991

inputsB
+conv1d_expanddims_1_readvariableop_resource:@ђ.
biasadd_readvariableop_resource:	ђ
identityѕбBiasAdd/ReadVariableOpб"Conv1D/ExpandDims_1/ReadVariableOp`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§        Ђ
Conv1D/ExpandDims
ExpandDimsinputsConv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:         @Њ
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:@ђ*
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : А
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:@ђГ
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:         ђ*
paddingSAME*
strides
Ђ
Conv1D/SqueezeSqueezeConv1D:output:0*
T0*,
_output_shapes
:         ђ*
squeeze_dims

§        s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0ѓ
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:         ђd
IdentityIdentityBiasAdd:output:0^NoOp*
T0*,
_output_shapes
:         ђё
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         @: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:         @
 
_user_specified_nameinputs
з
e
I__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_119967

inputs
identity[
	LeakyRelu	LeakyReluinputs*+
_output_shapes
:         @*
alpha%џЎЎ>c
IdentityIdentityLeakyRelu:activations:0*
T0*+
_output_shapes
:         @"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         @:S O
+
_output_shapes
:         @
 
_user_specified_nameinputs
Щ
q
D__inference_sampling_layer_call_and_return_conditional_losses_114193

inputs
inputs_1
identityѕ;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Л
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask=
Shape_1Shapeinputs*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask{
random_normal/shapePackstrided_slice:output:0strided_slice_1:output:0*
N*
T0*
_output_shapes
:W
random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    Y
random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?х
"random_normal/RandomStandardNormalRandomStandardNormalrandom_normal/shape:output:0*
T0*'
_output_shapes
:         *
dtype0*
seed▒ т)*
seed2╔Шќ
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*'
_output_shapes
:         |
random_normalAddV2random_normal/mul:z:0random_normal/mean:output:0*
T0*'
_output_shapes
:         J
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?V
mulMulmul/x:output:0inputs_1*
T0*'
_output_shapes
:         E
ExpExpmul:z:0*
T0*'
_output_shapes
:         Z
mul_1MulExp:y:0random_normal:z:0*
T0*'
_output_shapes
:         Q
addAddV2inputs	mul_1:z:0*
T0*'
_output_shapes
:         O
IdentityIdentityadd:z:0*
T0*'
_output_shapes
:         "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:         :         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs:OK
'
_output_shapes
:         
 
_user_specified_nameinputs
┌
Л
6__inference_batch_normalization_1_layer_call_fn_119903

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityѕбStatefulPartitionedCallІ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *Z
fURS
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_118598|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :                  @`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:                  @: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :                  @
 
_user_specified_nameinputs
К
ў
*__inference_z_log_var_layer_call_fn_120120

inputs
unknown:	ђ
	unknown_0:
identityѕбStatefulPartitionedCall┌
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *N
fIRG
E__inference_z_log_var_layer_call_and_return_conditional_losses_118843o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ђ: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
б
┤
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_120037

inputs0
!batchnorm_readvariableop_resource:	ђ4
%batchnorm_mul_readvariableop_resource:	ђ2
#batchnorm_readvariableop_1_resource:	ђ2
#batchnorm_readvariableop_2_resource:	ђ
identityѕбbatchnorm/ReadVariableOpбbatchnorm/ReadVariableOp_1бbatchnorm/ReadVariableOp_2бbatchnorm/mul/ReadVariableOpw
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:ђ*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:x
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:ђQ
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:ђ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:ђ*
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:ђq
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*5
_output_shapes#
!:                  ђ{
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0s
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:ђ{
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:ђ*
dtype0s
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:ђђ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*5
_output_shapes#
!:                  ђp
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*5
_output_shapes#
!:                  ђ║
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):                  ђ: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:] Y
5
_output_shapes#
!:                  ђ
 
_user_specified_nameinputs
╠	
э
E__inference_z_log_var_layer_call_and_return_conditional_losses_120130

inputs1
matmul_readvariableop_resource:	ђ-
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	ђ*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ђ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
Жя
Є
C__inference_encoder_layer_call_and_return_conditional_losses_119684

inputsH
2conv1d_conv1d_expanddims_1_readvariableop_resource: 4
&conv1d_biasadd_readvariableop_resource: I
;batch_normalization_assignmovingavg_readvariableop_resource: K
=batch_normalization_assignmovingavg_1_readvariableop_resource: G
9batch_normalization_batchnorm_mul_readvariableop_resource: C
5batch_normalization_batchnorm_readvariableop_resource: J
4conv1d_1_conv1d_expanddims_1_readvariableop_resource: @6
(conv1d_1_biasadd_readvariableop_resource:@K
=batch_normalization_1_assignmovingavg_readvariableop_resource:@M
?batch_normalization_1_assignmovingavg_1_readvariableop_resource:@I
;batch_normalization_1_batchnorm_mul_readvariableop_resource:@E
7batch_normalization_1_batchnorm_readvariableop_resource:@K
4conv1d_2_conv1d_expanddims_1_readvariableop_resource:@ђ7
(conv1d_2_biasadd_readvariableop_resource:	ђL
=batch_normalization_2_assignmovingavg_readvariableop_resource:	ђN
?batch_normalization_2_assignmovingavg_1_readvariableop_resource:	ђJ
;batch_normalization_2_batchnorm_mul_readvariableop_resource:	ђF
7batch_normalization_2_batchnorm_readvariableop_resource:	ђ<
)z_log_mean_matmul_readvariableop_resource:	ђ8
*z_log_mean_biasadd_readvariableop_resource:;
(z_log_var_matmul_readvariableop_resource:	ђ7
)z_log_var_biasadd_readvariableop_resource:
identity

identity_1

identity_2ѕб#batch_normalization/AssignMovingAvgб2batch_normalization/AssignMovingAvg/ReadVariableOpб%batch_normalization/AssignMovingAvg_1б4batch_normalization/AssignMovingAvg_1/ReadVariableOpб,batch_normalization/batchnorm/ReadVariableOpб0batch_normalization/batchnorm/mul/ReadVariableOpб%batch_normalization_1/AssignMovingAvgб4batch_normalization_1/AssignMovingAvg/ReadVariableOpб'batch_normalization_1/AssignMovingAvg_1б6batch_normalization_1/AssignMovingAvg_1/ReadVariableOpб.batch_normalization_1/batchnorm/ReadVariableOpб2batch_normalization_1/batchnorm/mul/ReadVariableOpб%batch_normalization_2/AssignMovingAvgб4batch_normalization_2/AssignMovingAvg/ReadVariableOpб'batch_normalization_2/AssignMovingAvg_1б6batch_normalization_2/AssignMovingAvg_1/ReadVariableOpб.batch_normalization_2/batchnorm/ReadVariableOpб2batch_normalization_2/batchnorm/mul/ReadVariableOpбconv1d/BiasAdd/ReadVariableOpб)conv1d/Conv1D/ExpandDims_1/ReadVariableOpбconv1d_1/BiasAdd/ReadVariableOpб+conv1d_1/Conv1D/ExpandDims_1/ReadVariableOpбconv1d_2/BiasAdd/ReadVariableOpб+conv1d_2/Conv1D/ExpandDims_1/ReadVariableOpб sampling/StatefulPartitionedCallб!z_log_mean/BiasAdd/ReadVariableOpб z_log_mean/MatMul/ReadVariableOpб z_log_var/BiasAdd/ReadVariableOpбz_log_var/MatMul/ReadVariableOpg
conv1d/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§        Ј
conv1d/Conv1D/ExpandDims
ExpandDimsinputs%conv1d/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:         а
)conv1d/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp2conv1d_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype0`
conv1d/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : х
conv1d/Conv1D/ExpandDims_1
ExpandDims1conv1d/Conv1D/ExpandDims_1/ReadVariableOp:value:0'conv1d/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: ┴
conv1d/Conv1DConv2D!conv1d/Conv1D/ExpandDims:output:0#conv1d/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:          *
paddingSAME*
strides
ј
conv1d/Conv1D/SqueezeSqueezeconv1d/Conv1D:output:0*
T0*+
_output_shapes
:          *
squeeze_dims

§        ђ
conv1d/BiasAdd/ReadVariableOpReadVariableOp&conv1d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0ќ
conv1d/BiasAddBiasAddconv1d/Conv1D/Squeeze:output:0%conv1d/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:          Ѓ
2batch_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       ╝
 batch_normalization/moments/meanMeanconv1d/BiasAdd:output:0;batch_normalization/moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
: *
	keep_dims(љ
(batch_normalization/moments/StopGradientStopGradient)batch_normalization/moments/mean:output:0*
T0*"
_output_shapes
: ─
-batch_normalization/moments/SquaredDifferenceSquaredDifferenceconv1d/BiasAdd:output:01batch_normalization/moments/StopGradient:output:0*
T0*+
_output_shapes
:          Є
6batch_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       я
$batch_normalization/moments/varianceMean1batch_normalization/moments/SquaredDifference:z:0?batch_normalization/moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
: *
	keep_dims(ќ
#batch_normalization/moments/SqueezeSqueeze)batch_normalization/moments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 ю
%batch_normalization/moments/Squeeze_1Squeeze-batch_normalization/moments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 n
)batch_normalization/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<ф
2batch_normalization/AssignMovingAvg/ReadVariableOpReadVariableOp;batch_normalization_assignmovingavg_readvariableop_resource*
_output_shapes
: *
dtype0й
'batch_normalization/AssignMovingAvg/subSub:batch_normalization/AssignMovingAvg/ReadVariableOp:value:0,batch_normalization/moments/Squeeze:output:0*
T0*
_output_shapes
: ┤
'batch_normalization/AssignMovingAvg/mulMul+batch_normalization/AssignMovingAvg/sub:z:02batch_normalization/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
: Ч
#batch_normalization/AssignMovingAvgAssignSubVariableOp;batch_normalization_assignmovingavg_readvariableop_resource+batch_normalization/AssignMovingAvg/mul:z:03^batch_normalization/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0p
+batch_normalization/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<«
4batch_normalization/AssignMovingAvg_1/ReadVariableOpReadVariableOp=batch_normalization_assignmovingavg_1_readvariableop_resource*
_output_shapes
: *
dtype0├
)batch_normalization/AssignMovingAvg_1/subSub<batch_normalization/AssignMovingAvg_1/ReadVariableOp:value:0.batch_normalization/moments/Squeeze_1:output:0*
T0*
_output_shapes
: ║
)batch_normalization/AssignMovingAvg_1/mulMul-batch_normalization/AssignMovingAvg_1/sub:z:04batch_normalization/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
: ё
%batch_normalization/AssignMovingAvg_1AssignSubVariableOp=batch_normalization_assignmovingavg_1_readvariableop_resource-batch_normalization/AssignMovingAvg_1/mul:z:05^batch_normalization/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0h
#batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:Г
!batch_normalization/batchnorm/addAddV2.batch_normalization/moments/Squeeze_1:output:0,batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes
: x
#batch_normalization/batchnorm/RsqrtRsqrt%batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes
: д
0batch_normalization/batchnorm/mul/ReadVariableOpReadVariableOp9batch_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype0░
!batch_normalization/batchnorm/mulMul'batch_normalization/batchnorm/Rsqrt:y:08batch_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: а
#batch_normalization/batchnorm/mul_1Mulconv1d/BiasAdd:output:0%batch_normalization/batchnorm/mul:z:0*
T0*+
_output_shapes
:          ц
#batch_normalization/batchnorm/mul_2Mul,batch_normalization/moments/Squeeze:output:0%batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes
: ъ
,batch_normalization/batchnorm/ReadVariableOpReadVariableOp5batch_normalization_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype0г
!batch_normalization/batchnorm/subSub4batch_normalization/batchnorm/ReadVariableOp:value:0'batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes
: ▓
#batch_normalization/batchnorm/add_1AddV2'batch_normalization/batchnorm/mul_1:z:0%batch_normalization/batchnorm/sub:z:0*
T0*+
_output_shapes
:          ѕ
leaky_re_lu/LeakyRelu	LeakyRelu'batch_normalization/batchnorm/add_1:z:0*+
_output_shapes
:          *
alpha%џЎЎ>i
conv1d_1/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§        ░
conv1d_1/Conv1D/ExpandDims
ExpandDims#leaky_re_lu/LeakyRelu:activations:0'conv1d_1/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:          ц
+conv1d_1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: @*
dtype0b
 conv1d_1/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ╗
conv1d_1/Conv1D/ExpandDims_1
ExpandDims3conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp:value:0)conv1d_1/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: @К
conv1d_1/Conv1DConv2D#conv1d_1/Conv1D/ExpandDims:output:0%conv1d_1/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
њ
conv1d_1/Conv1D/SqueezeSqueezeconv1d_1/Conv1D:output:0*
T0*+
_output_shapes
:         @*
squeeze_dims

§        ё
conv1d_1/BiasAdd/ReadVariableOpReadVariableOp(conv1d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0ю
conv1d_1/BiasAddBiasAdd conv1d_1/Conv1D/Squeeze:output:0'conv1d_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         @Ё
4batch_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       ┬
"batch_normalization_1/moments/meanMeanconv1d_1/BiasAdd:output:0=batch_normalization_1/moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(ћ
*batch_normalization_1/moments/StopGradientStopGradient+batch_normalization_1/moments/mean:output:0*
T0*"
_output_shapes
:@╩
/batch_normalization_1/moments/SquaredDifferenceSquaredDifferenceconv1d_1/BiasAdd:output:03batch_normalization_1/moments/StopGradient:output:0*
T0*+
_output_shapes
:         @Ѕ
8batch_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       С
&batch_normalization_1/moments/varianceMean3batch_normalization_1/moments/SquaredDifference:z:0Abatch_normalization_1/moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(џ
%batch_normalization_1/moments/SqueezeSqueeze+batch_normalization_1/moments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 а
'batch_normalization_1/moments/Squeeze_1Squeeze/batch_normalization_1/moments/variance:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 p
+batch_normalization_1/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<«
4batch_normalization_1/AssignMovingAvg/ReadVariableOpReadVariableOp=batch_normalization_1_assignmovingavg_readvariableop_resource*
_output_shapes
:@*
dtype0├
)batch_normalization_1/AssignMovingAvg/subSub<batch_normalization_1/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_1/moments/Squeeze:output:0*
T0*
_output_shapes
:@║
)batch_normalization_1/AssignMovingAvg/mulMul-batch_normalization_1/AssignMovingAvg/sub:z:04batch_normalization_1/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:@ё
%batch_normalization_1/AssignMovingAvgAssignSubVariableOp=batch_normalization_1_assignmovingavg_readvariableop_resource-batch_normalization_1/AssignMovingAvg/mul:z:05^batch_normalization_1/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0r
-batch_normalization_1/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<▓
6batch_normalization_1/AssignMovingAvg_1/ReadVariableOpReadVariableOp?batch_normalization_1_assignmovingavg_1_readvariableop_resource*
_output_shapes
:@*
dtype0╔
+batch_normalization_1/AssignMovingAvg_1/subSub>batch_normalization_1/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_1/moments/Squeeze_1:output:0*
T0*
_output_shapes
:@└
+batch_normalization_1/AssignMovingAvg_1/mulMul/batch_normalization_1/AssignMovingAvg_1/sub:z:06batch_normalization_1/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:@ї
'batch_normalization_1/AssignMovingAvg_1AssignSubVariableOp?batch_normalization_1_assignmovingavg_1_readvariableop_resource/batch_normalization_1/AssignMovingAvg_1/mul:z:07^batch_normalization_1/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0j
%batch_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:│
#batch_normalization_1/batchnorm/addAddV20batch_normalization_1/moments/Squeeze_1:output:0.batch_normalization_1/batchnorm/add/y:output:0*
T0*
_output_shapes
:@|
%batch_normalization_1/batchnorm/RsqrtRsqrt'batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes
:@ф
2batch_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0Х
#batch_normalization_1/batchnorm/mulMul)batch_normalization_1/batchnorm/Rsqrt:y:0:batch_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@д
%batch_normalization_1/batchnorm/mul_1Mulconv1d_1/BiasAdd:output:0'batch_normalization_1/batchnorm/mul:z:0*
T0*+
_output_shapes
:         @ф
%batch_normalization_1/batchnorm/mul_2Mul.batch_normalization_1/moments/Squeeze:output:0'batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes
:@б
.batch_normalization_1/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_1_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0▓
#batch_normalization_1/batchnorm/subSub6batch_normalization_1/batchnorm/ReadVariableOp:value:0)batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@И
%batch_normalization_1/batchnorm/add_1AddV2)batch_normalization_1/batchnorm/mul_1:z:0'batch_normalization_1/batchnorm/sub:z:0*
T0*+
_output_shapes
:         @ї
leaky_re_lu_1/LeakyRelu	LeakyRelu)batch_normalization_1/batchnorm/add_1:z:0*+
_output_shapes
:         @*
alpha%џЎЎ>i
conv1d_2/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§        ▓
conv1d_2/Conv1D/ExpandDims
ExpandDims%leaky_re_lu_1/LeakyRelu:activations:0'conv1d_2/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:         @Ц
+conv1d_2/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_2_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:@ђ*
dtype0b
 conv1d_2/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ╝
conv1d_2/Conv1D/ExpandDims_1
ExpandDims3conv1d_2/Conv1D/ExpandDims_1/ReadVariableOp:value:0)conv1d_2/Conv1D/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:@ђ╚
conv1d_2/Conv1DConv2D#conv1d_2/Conv1D/ExpandDims:output:0%conv1d_2/Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:         ђ*
paddingSAME*
strides
Њ
conv1d_2/Conv1D/SqueezeSqueezeconv1d_2/Conv1D:output:0*
T0*,
_output_shapes
:         ђ*
squeeze_dims

§        Ё
conv1d_2/BiasAdd/ReadVariableOpReadVariableOp(conv1d_2_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0Ю
conv1d_2/BiasAddBiasAdd conv1d_2/Conv1D/Squeeze:output:0'conv1d_2/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:         ђЁ
4batch_normalization_2/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       ├
"batch_normalization_2/moments/meanMeanconv1d_2/BiasAdd:output:0=batch_normalization_2/moments/mean/reduction_indices:output:0*
T0*#
_output_shapes
:ђ*
	keep_dims(Ћ
*batch_normalization_2/moments/StopGradientStopGradient+batch_normalization_2/moments/mean:output:0*
T0*#
_output_shapes
:ђ╦
/batch_normalization_2/moments/SquaredDifferenceSquaredDifferenceconv1d_2/BiasAdd:output:03batch_normalization_2/moments/StopGradient:output:0*
T0*,
_output_shapes
:         ђЅ
8batch_normalization_2/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       т
&batch_normalization_2/moments/varianceMean3batch_normalization_2/moments/SquaredDifference:z:0Abatch_normalization_2/moments/variance/reduction_indices:output:0*
T0*#
_output_shapes
:ђ*
	keep_dims(Џ
%batch_normalization_2/moments/SqueezeSqueeze+batch_normalization_2/moments/mean:output:0*
T0*
_output_shapes	
:ђ*
squeeze_dims
 А
'batch_normalization_2/moments/Squeeze_1Squeeze/batch_normalization_2/moments/variance:output:0*
T0*
_output_shapes	
:ђ*
squeeze_dims
 p
+batch_normalization_2/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<»
4batch_normalization_2/AssignMovingAvg/ReadVariableOpReadVariableOp=batch_normalization_2_assignmovingavg_readvariableop_resource*
_output_shapes	
:ђ*
dtype0─
)batch_normalization_2/AssignMovingAvg/subSub<batch_normalization_2/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_2/moments/Squeeze:output:0*
T0*
_output_shapes	
:ђ╗
)batch_normalization_2/AssignMovingAvg/mulMul-batch_normalization_2/AssignMovingAvg/sub:z:04batch_normalization_2/AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:ђё
%batch_normalization_2/AssignMovingAvgAssignSubVariableOp=batch_normalization_2_assignmovingavg_readvariableop_resource-batch_normalization_2/AssignMovingAvg/mul:z:05^batch_normalization_2/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0r
-batch_normalization_2/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<│
6batch_normalization_2/AssignMovingAvg_1/ReadVariableOpReadVariableOp?batch_normalization_2_assignmovingavg_1_readvariableop_resource*
_output_shapes	
:ђ*
dtype0╩
+batch_normalization_2/AssignMovingAvg_1/subSub>batch_normalization_2/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_2/moments/Squeeze_1:output:0*
T0*
_output_shapes	
:ђ┴
+batch_normalization_2/AssignMovingAvg_1/mulMul/batch_normalization_2/AssignMovingAvg_1/sub:z:06batch_normalization_2/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:ђї
'batch_normalization_2/AssignMovingAvg_1AssignSubVariableOp?batch_normalization_2_assignmovingavg_1_readvariableop_resource/batch_normalization_2/AssignMovingAvg_1/mul:z:07^batch_normalization_2/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0j
%batch_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:┤
#batch_normalization_2/batchnorm/addAddV20batch_normalization_2/moments/Squeeze_1:output:0.batch_normalization_2/batchnorm/add/y:output:0*
T0*
_output_shapes	
:ђ}
%batch_normalization_2/batchnorm/RsqrtRsqrt'batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes	
:ђФ
2batch_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes	
:ђ*
dtype0и
#batch_normalization_2/batchnorm/mulMul)batch_normalization_2/batchnorm/Rsqrt:y:0:batch_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:ђД
%batch_normalization_2/batchnorm/mul_1Mulconv1d_2/BiasAdd:output:0'batch_normalization_2/batchnorm/mul:z:0*
T0*,
_output_shapes
:         ђФ
%batch_normalization_2/batchnorm/mul_2Mul.batch_normalization_2/moments/Squeeze:output:0'batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes	
:ђБ
.batch_normalization_2/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_2_batchnorm_readvariableop_resource*
_output_shapes	
:ђ*
dtype0│
#batch_normalization_2/batchnorm/subSub6batch_normalization_2/batchnorm/ReadVariableOp:value:0)batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:ђ╣
%batch_normalization_2/batchnorm/add_1AddV2)batch_normalization_2/batchnorm/mul_1:z:0'batch_normalization_2/batchnorm/sub:z:0*
T0*,
_output_shapes
:         ђЇ
leaky_re_lu_2/LeakyRelu	LeakyRelu)batch_normalization_2/batchnorm/add_1:z:0*,
_output_shapes
:         ђ*
alpha%џЎЎ>^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ї
flatten/ReshapeReshape%leaky_re_lu_2/LeakyRelu:activations:0flatten/Const:output:0*
T0*(
_output_shapes
:         ђІ
 z_log_mean/MatMul/ReadVariableOpReadVariableOp)z_log_mean_matmul_readvariableop_resource*
_output_shapes
:	ђ*
dtype0Љ
z_log_mean/MatMulMatMulflatten/Reshape:output:0(z_log_mean/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         ѕ
!z_log_mean/BiasAdd/ReadVariableOpReadVariableOp*z_log_mean_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ќ
z_log_mean/BiasAddBiasAddz_log_mean/MatMul:product:0)z_log_mean/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         Ѕ
z_log_var/MatMul/ReadVariableOpReadVariableOp(z_log_var_matmul_readvariableop_resource*
_output_shapes
:	ђ*
dtype0Ј
z_log_var/MatMulMatMulflatten/Reshape:output:0'z_log_var/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         є
 z_log_var/BiasAdd/ReadVariableOpReadVariableOp)z_log_var_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0ћ
z_log_var/BiasAddBiasAddz_log_var/MatMul:product:0(z_log_var/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         ▀
 sampling/StatefulPartitionedCallStatefulPartitionedCallz_log_mean/BiasAdd:output:0z_log_var/BiasAdd:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *2
f-R+
)__inference_restored_function_body_118440j
IdentityIdentityz_log_mean/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         k

Identity_1Identityz_log_var/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         z

Identity_2Identity)sampling/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         ═

NoOpNoOp$^batch_normalization/AssignMovingAvg3^batch_normalization/AssignMovingAvg/ReadVariableOp&^batch_normalization/AssignMovingAvg_15^batch_normalization/AssignMovingAvg_1/ReadVariableOp-^batch_normalization/batchnorm/ReadVariableOp1^batch_normalization/batchnorm/mul/ReadVariableOp&^batch_normalization_1/AssignMovingAvg5^batch_normalization_1/AssignMovingAvg/ReadVariableOp(^batch_normalization_1/AssignMovingAvg_17^batch_normalization_1/AssignMovingAvg_1/ReadVariableOp/^batch_normalization_1/batchnorm/ReadVariableOp3^batch_normalization_1/batchnorm/mul/ReadVariableOp&^batch_normalization_2/AssignMovingAvg5^batch_normalization_2/AssignMovingAvg/ReadVariableOp(^batch_normalization_2/AssignMovingAvg_17^batch_normalization_2/AssignMovingAvg_1/ReadVariableOp/^batch_normalization_2/batchnorm/ReadVariableOp3^batch_normalization_2/batchnorm/mul/ReadVariableOp^conv1d/BiasAdd/ReadVariableOp*^conv1d/Conv1D/ExpandDims_1/ReadVariableOp ^conv1d_1/BiasAdd/ReadVariableOp,^conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp ^conv1d_2/BiasAdd/ReadVariableOp,^conv1d_2/Conv1D/ExpandDims_1/ReadVariableOp!^sampling/StatefulPartitionedCall"^z_log_mean/BiasAdd/ReadVariableOp!^z_log_mean/MatMul/ReadVariableOp!^z_log_var/BiasAdd/ReadVariableOp ^z_log_var/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:         : : : : : : : : : : : : : : : : : : : : : : 2J
#batch_normalization/AssignMovingAvg#batch_normalization/AssignMovingAvg2h
2batch_normalization/AssignMovingAvg/ReadVariableOp2batch_normalization/AssignMovingAvg/ReadVariableOp2N
%batch_normalization/AssignMovingAvg_1%batch_normalization/AssignMovingAvg_12l
4batch_normalization/AssignMovingAvg_1/ReadVariableOp4batch_normalization/AssignMovingAvg_1/ReadVariableOp2\
,batch_normalization/batchnorm/ReadVariableOp,batch_normalization/batchnorm/ReadVariableOp2d
0batch_normalization/batchnorm/mul/ReadVariableOp0batch_normalization/batchnorm/mul/ReadVariableOp2N
%batch_normalization_1/AssignMovingAvg%batch_normalization_1/AssignMovingAvg2l
4batch_normalization_1/AssignMovingAvg/ReadVariableOp4batch_normalization_1/AssignMovingAvg/ReadVariableOp2R
'batch_normalization_1/AssignMovingAvg_1'batch_normalization_1/AssignMovingAvg_12p
6batch_normalization_1/AssignMovingAvg_1/ReadVariableOp6batch_normalization_1/AssignMovingAvg_1/ReadVariableOp2`
.batch_normalization_1/batchnorm/ReadVariableOp.batch_normalization_1/batchnorm/ReadVariableOp2h
2batch_normalization_1/batchnorm/mul/ReadVariableOp2batch_normalization_1/batchnorm/mul/ReadVariableOp2N
%batch_normalization_2/AssignMovingAvg%batch_normalization_2/AssignMovingAvg2l
4batch_normalization_2/AssignMovingAvg/ReadVariableOp4batch_normalization_2/AssignMovingAvg/ReadVariableOp2R
'batch_normalization_2/AssignMovingAvg_1'batch_normalization_2/AssignMovingAvg_12p
6batch_normalization_2/AssignMovingAvg_1/ReadVariableOp6batch_normalization_2/AssignMovingAvg_1/ReadVariableOp2`
.batch_normalization_2/batchnorm/ReadVariableOp.batch_normalization_2/batchnorm/ReadVariableOp2h
2batch_normalization_2/batchnorm/mul/ReadVariableOp2batch_normalization_2/batchnorm/mul/ReadVariableOp2>
conv1d/BiasAdd/ReadVariableOpconv1d/BiasAdd/ReadVariableOp2V
)conv1d/Conv1D/ExpandDims_1/ReadVariableOp)conv1d/Conv1D/ExpandDims_1/ReadVariableOp2B
conv1d_1/BiasAdd/ReadVariableOpconv1d_1/BiasAdd/ReadVariableOp2Z
+conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp+conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp2B
conv1d_2/BiasAdd/ReadVariableOpconv1d_2/BiasAdd/ReadVariableOp2Z
+conv1d_2/Conv1D/ExpandDims_1/ReadVariableOp+conv1d_2/Conv1D/ExpandDims_1/ReadVariableOp2D
 sampling/StatefulPartitionedCall sampling/StatefulPartitionedCall2F
!z_log_mean/BiasAdd/ReadVariableOp!z_log_mean/BiasAdd/ReadVariableOp2D
 z_log_mean/MatMul/ReadVariableOp z_log_mean/MatMul/ReadVariableOp2D
 z_log_var/BiasAdd/ReadVariableOp z_log_var/BiasAdd/ReadVariableOp2B
z_log_var/MatMul/ReadVariableOpz_log_var/MatMul/ReadVariableOp:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
џ&
Ь
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_120071

inputs6
'assignmovingavg_readvariableop_resource:	ђ8
)assignmovingavg_1_readvariableop_resource:	ђ4
%batchnorm_mul_readvariableop_resource:	ђ0
!batchnorm_readvariableop_resource:	ђ
identityѕбAssignMovingAvgбAssignMovingAvg/ReadVariableOpбAssignMovingAvg_1б AssignMovingAvg_1/ReadVariableOpбbatchnorm/ReadVariableOpбbatchnorm/mul/ReadVariableOpo
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       ё
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*#
_output_shapes
:ђ*
	keep_dims(i
moments/StopGradientStopGradientmoments/mean:output:0*
T0*#
_output_shapes
:ђЋ
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*5
_output_shapes#
!:                  ђs
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       Б
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*#
_output_shapes
:ђ*
	keep_dims(o
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:ђ*
squeeze_dims
 u
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:ђ*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<Ѓ
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:ђ*
dtype0ѓ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:ђy
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:ђг
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
О#<Є
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:ђ*
dtype0ѕ
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:ђ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:ђ┤
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:r
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:ђQ
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:ђ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:ђ*
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:ђq
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*5
_output_shapes#
!:                  ђi
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:ђw
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:ђ*
dtype0q
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:ђђ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*5
_output_shapes#
!:                  ђp
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*5
_output_shapes#
!:                  ђЖ
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):                  ђ: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:] Y
5
_output_shapes#
!:                  ђ
 
_user_specified_nameinputs
│
H
,__inference_leaky_re_lu_layer_call_fn_119848

inputs
identityХ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:          * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *P
fKRI
G__inference_leaky_re_lu_layer_call_and_return_conditional_losses_118733d
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:          "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:          :S O
+
_output_shapes
:          
 
_user_specified_nameinputs
╠	
э
E__inference_z_log_var_layer_call_and_return_conditional_losses_118843

inputs1
matmul_readvariableop_resource:	ђ-
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	ђ*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ђ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
№
Њ
D__inference_conv1d_1_layer_call_and_return_conditional_losses_119877

inputsA
+conv1d_expanddims_1_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identityѕбBiasAdd/ReadVariableOpб"Conv1D/ExpandDims_1/ReadVariableOp`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§        Ђ
Conv1D/ExpandDims
ExpandDimsinputsConv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:          њ
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: @*
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : а
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: @г
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
ђ
Conv1D/SqueezeSqueezeConv1D:output:0*
T0*+
_output_shapes
:         @*
squeeze_dims

§        r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ђ
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         @c
IdentityIdentityBiasAdd:output:0^NoOp*
T0*+
_output_shapes
:         @ё
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:          : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:          
 
_user_specified_nameinputs
┌
ю
)__inference_conv1d_2_layer_call_fn_119976

inputs
unknown:@ђ
	unknown_0:	ђ
identityѕбStatefulPartitionedCallя
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ђ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_conv1d_2_layer_call_and_return_conditional_losses_118787t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:         ђ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         @: : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:         @
 
_user_specified_nameinputs
Р
Н
6__inference_batch_normalization_2_layer_call_fn_120017

inputs
unknown:	ђ
	unknown_0:	ђ
	unknown_1:	ђ
	unknown_2:	ђ
identityѕбStatefulPartitionedCallї
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:                  ђ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *Z
fURS
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_118680}
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:                  ђ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):                  ђ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:                  ђ
 
_user_specified_nameinputs
э
r
)__inference_sampling_layer_call_fn_114199
inputs_0
inputs_1
identityѕбStatefulPartitionedCall¤
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *M
fHRF
D__inference_sampling_layer_call_and_return_conditional_losses_114193`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:         :         22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:         
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:         
"
_user_specified_name
inputs/1
С
Н
6__inference_batch_normalization_2_layer_call_fn_120004

inputs
unknown:	ђ
	unknown_0:	ђ
	unknown_1:	ђ
	unknown_2:	ђ
identityѕбStatefulPartitionedCallј
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:                  ђ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *Z
fURS
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_118633}
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:                  ђ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):                  ђ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:                  ђ
 
_user_specified_nameinputs
┐
_
C__inference_flatten_layer_call_and_return_conditional_losses_118815

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"       ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:         ђY
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:         ђ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ђ:T P
,
_output_shapes
:         ђ
 
_user_specified_nameinputs
§%
Ж
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_118598

inputs5
'assignmovingavg_readvariableop_resource:@7
)assignmovingavg_1_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@/
!batchnorm_readvariableop_resource:@
identityѕбAssignMovingAvgбAssignMovingAvg/ReadVariableOpбAssignMovingAvg_1б AssignMovingAvg_1/ReadVariableOpбbatchnorm/ReadVariableOpбbatchnorm/mul/ReadVariableOpo
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       Ѓ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(h
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
:@ћ
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*4
_output_shapes"
 :                  @s
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       б
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(n
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 t
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<ѓ
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:@*
dtype0Ђ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:@x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:@г
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
О#<є
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:@*
dtype0Є
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:@~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:@┤
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@p
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :                  @h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:@v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :                  @o
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :                  @Ж
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:                  @: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :                  @
 
_user_specified_nameinputs
ч%
У
O__inference_batch_normalization_layer_call_and_return_conditional_losses_118516

inputs5
'assignmovingavg_readvariableop_resource: 7
)assignmovingavg_1_readvariableop_resource: 3
%batchnorm_mul_readvariableop_resource: /
!batchnorm_readvariableop_resource: 
identityѕбAssignMovingAvgбAssignMovingAvg/ReadVariableOpбAssignMovingAvg_1б AssignMovingAvg_1/ReadVariableOpбbatchnorm/ReadVariableOpбbatchnorm/mul/ReadVariableOpo
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       Ѓ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
: *
	keep_dims(h
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
: ћ
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*4
_output_shapes"
 :                   s
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       б
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
: *
	keep_dims(n
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 t
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<ѓ
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
: *
dtype0Ђ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
: x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
: г
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
О#<є
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
: *
dtype0Є
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
: ~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
: ┤
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:q
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
: p
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :                   h
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
: 
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :                   o
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :                   Ж
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:                   : : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :                   
 
_user_specified_nameinputs
и
J
.__inference_leaky_re_lu_1_layer_call_fn_119962

inputs
identityИ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *R
fMRK
I__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_118770d
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:         @"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         @:S O
+
_output_shapes
:         @
 
_user_specified_nameinputs"█L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*«
serving_defaultџ
?
input_14
serving_default_input_1:0         <
sampling0
StatefulPartitionedCall:0         >

z_log_mean0
StatefulPartitionedCall:1         =
	z_log_var0
StatefulPartitionedCall:2         tensorflow/serving/predict:Зж
Ѓ
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer-6
layer_with_weights-4
layer-7
	layer_with_weights-5
	layer-8

layer-9
layer-10
layer_with_weights-6
layer-11
layer_with_weights-7
layer-12
layer-13

signatures
#_self_saveable_object_factories
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature"
_tf_keras_network
D
#_self_saveable_object_factories"
_tf_keras_input_layer
Я

kernel
bias
#_self_saveable_object_factories
	variables
trainable_variables
regularization_losses
	keras_api
 __call__
*!&call_and_return_all_conditional_losses"
_tf_keras_layer
Ј
"axis
	#gamma
$beta
%moving_mean
&moving_variance
#'_self_saveable_object_factories
(	variables
)trainable_variables
*regularization_losses
+	keras_api
,__call__
*-&call_and_return_all_conditional_losses"
_tf_keras_layer
╩
#._self_saveable_object_factories
/	variables
0trainable_variables
1regularization_losses
2	keras_api
3__call__
*4&call_and_return_all_conditional_losses"
_tf_keras_layer
Я

5kernel
6bias
#7_self_saveable_object_factories
8	variables
9trainable_variables
:regularization_losses
;	keras_api
<__call__
*=&call_and_return_all_conditional_losses"
_tf_keras_layer
Ј
>axis
	?gamma
@beta
Amoving_mean
Bmoving_variance
#C_self_saveable_object_factories
D	variables
Etrainable_variables
Fregularization_losses
G	keras_api
H__call__
*I&call_and_return_all_conditional_losses"
_tf_keras_layer
╩
#J_self_saveable_object_factories
K	variables
Ltrainable_variables
Mregularization_losses
N	keras_api
O__call__
*P&call_and_return_all_conditional_losses"
_tf_keras_layer
Я

Qkernel
Rbias
#S_self_saveable_object_factories
T	variables
Utrainable_variables
Vregularization_losses
W	keras_api
X__call__
*Y&call_and_return_all_conditional_losses"
_tf_keras_layer
Ј
Zaxis
	[gamma
\beta
]moving_mean
^moving_variance
#__self_saveable_object_factories
`	variables
atrainable_variables
bregularization_losses
c	keras_api
d__call__
*e&call_and_return_all_conditional_losses"
_tf_keras_layer
╩
#f_self_saveable_object_factories
g	variables
htrainable_variables
iregularization_losses
j	keras_api
k__call__
*l&call_and_return_all_conditional_losses"
_tf_keras_layer
╩
#m_self_saveable_object_factories
n	variables
otrainable_variables
pregularization_losses
q	keras_api
r__call__
*s&call_and_return_all_conditional_losses"
_tf_keras_layer
Я

tkernel
ubias
#v_self_saveable_object_factories
w	variables
xtrainable_variables
yregularization_losses
z	keras_api
{__call__
*|&call_and_return_all_conditional_losses"
_tf_keras_layer
Т

}kernel
~bias
#_self_saveable_object_factories
ђ	variables
Ђtrainable_variables
ѓregularization_losses
Ѓ	keras_api
ё__call__
+Ё&call_and_return_all_conditional_losses"
_tf_keras_layer
Л
$є_self_saveable_object_factories
Є	variables
ѕtrainable_variables
Ѕregularization_losses
і	keras_api
І__call__
+ї&call_and_return_all_conditional_losses"
_tf_keras_layer
-
Їserving_default"
signature_map
 "
trackable_dict_wrapper
к
0
1
#2
$3
%4
&5
56
67
?8
@9
A10
B11
Q12
R13
[14
\15
]16
^17
t18
u19
}20
~21"
trackable_list_wrapper
ќ
0
1
#2
$3
54
65
?6
@7
Q8
R9
[10
\11
t12
u13
}14
~15"
trackable_list_wrapper
 "
trackable_list_wrapper
¤
јnon_trainable_variables
Јlayers
љmetrics
 Љlayer_regularization_losses
њlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
Ь2в
(__inference_encoder_layer_call_fn_118904
(__inference_encoder_layer_call_fn_119379
(__inference_encoder_layer_call_fn_119432
(__inference_encoder_layer_call_fn_119200└
и▓│
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
┌2О
C__inference_encoder_layer_call_and_return_conditional_losses_119537
C__inference_encoder_layer_call_and_return_conditional_losses_119684
C__inference_encoder_layer_call_and_return_conditional_losses_119263
C__inference_encoder_layer_call_and_return_conditional_losses_119326└
и▓│
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
╠B╔
!__inference__wrapped_model_118445input_1"ў
Љ▓Ї
FullArgSpec
argsџ 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
 "
trackable_dict_wrapper
#:! 2conv1d/kernel
: 2conv1d/bias
 "
trackable_dict_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
Њnon_trainable_variables
ћlayers
Ћmetrics
 ќlayer_regularization_losses
Ќlayer_metrics
	variables
trainable_variables
regularization_losses
 __call__
*!&call_and_return_all_conditional_losses
&!"call_and_return_conditional_losses"
_generic_user_object
Л2╬
'__inference_conv1d_layer_call_fn_119748б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
В2ж
B__inference_conv1d_layer_call_and_return_conditional_losses_119763б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
 "
trackable_list_wrapper
':% 2batch_normalization/gamma
&:$ 2batch_normalization/beta
/:-  (2batch_normalization/moving_mean
3:1  (2#batch_normalization/moving_variance
 "
trackable_dict_wrapper
<
#0
$1
%2
&3"
trackable_list_wrapper
.
#0
$1"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
ўnon_trainable_variables
Ўlayers
џmetrics
 Џlayer_regularization_losses
юlayer_metrics
(	variables
)trainable_variables
*regularization_losses
,__call__
*-&call_and_return_all_conditional_losses
&-"call_and_return_conditional_losses"
_generic_user_object
д2Б
4__inference_batch_normalization_layer_call_fn_119776
4__inference_batch_normalization_layer_call_fn_119789┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
▄2┘
O__inference_batch_normalization_layer_call_and_return_conditional_losses_119809
O__inference_batch_normalization_layer_call_and_return_conditional_losses_119843┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
▓
Юnon_trainable_variables
ъlayers
Ъmetrics
 аlayer_regularization_losses
Аlayer_metrics
/	variables
0trainable_variables
1regularization_losses
3__call__
*4&call_and_return_all_conditional_losses
&4"call_and_return_conditional_losses"
_generic_user_object
о2М
,__inference_leaky_re_lu_layer_call_fn_119848б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ы2Ь
G__inference_leaky_re_lu_layer_call_and_return_conditional_losses_119853б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
%:# @2conv1d_1/kernel
:@2conv1d_1/bias
 "
trackable_dict_wrapper
.
50
61"
trackable_list_wrapper
.
50
61"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
бnon_trainable_variables
Бlayers
цmetrics
 Цlayer_regularization_losses
дlayer_metrics
8	variables
9trainable_variables
:regularization_losses
<__call__
*=&call_and_return_all_conditional_losses
&="call_and_return_conditional_losses"
_generic_user_object
М2л
)__inference_conv1d_1_layer_call_fn_119862б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Ь2в
D__inference_conv1d_1_layer_call_and_return_conditional_losses_119877б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
 "
trackable_list_wrapper
):'@2batch_normalization_1/gamma
(:&@2batch_normalization_1/beta
1:/@ (2!batch_normalization_1/moving_mean
5:3@ (2%batch_normalization_1/moving_variance
 "
trackable_dict_wrapper
<
?0
@1
A2
B3"
trackable_list_wrapper
.
?0
@1"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
Дnon_trainable_variables
еlayers
Еmetrics
 фlayer_regularization_losses
Фlayer_metrics
D	variables
Etrainable_variables
Fregularization_losses
H__call__
*I&call_and_return_all_conditional_losses
&I"call_and_return_conditional_losses"
_generic_user_object
ф2Д
6__inference_batch_normalization_1_layer_call_fn_119890
6__inference_batch_normalization_1_layer_call_fn_119903┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
Я2П
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_119923
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_119957┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
▓
гnon_trainable_variables
Гlayers
«metrics
 »layer_regularization_losses
░layer_metrics
K	variables
Ltrainable_variables
Mregularization_losses
O__call__
*P&call_and_return_all_conditional_losses
&P"call_and_return_conditional_losses"
_generic_user_object
п2Н
.__inference_leaky_re_lu_1_layer_call_fn_119962б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
з2­
I__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_119967б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
&:$@ђ2conv1d_2/kernel
:ђ2conv1d_2/bias
 "
trackable_dict_wrapper
.
Q0
R1"
trackable_list_wrapper
.
Q0
R1"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
▒non_trainable_variables
▓layers
│metrics
 ┤layer_regularization_losses
хlayer_metrics
T	variables
Utrainable_variables
Vregularization_losses
X__call__
*Y&call_and_return_all_conditional_losses
&Y"call_and_return_conditional_losses"
_generic_user_object
М2л
)__inference_conv1d_2_layer_call_fn_119976б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Ь2в
D__inference_conv1d_2_layer_call_and_return_conditional_losses_119991б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
 "
trackable_list_wrapper
*:(ђ2batch_normalization_2/gamma
):'ђ2batch_normalization_2/beta
2:0ђ (2!batch_normalization_2/moving_mean
6:4ђ (2%batch_normalization_2/moving_variance
 "
trackable_dict_wrapper
<
[0
\1
]2
^3"
trackable_list_wrapper
.
[0
\1"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
Хnon_trainable_variables
иlayers
Иmetrics
 ╣layer_regularization_losses
║layer_metrics
`	variables
atrainable_variables
bregularization_losses
d__call__
*e&call_and_return_all_conditional_losses
&e"call_and_return_conditional_losses"
_generic_user_object
ф2Д
6__inference_batch_normalization_2_layer_call_fn_120004
6__inference_batch_normalization_2_layer_call_fn_120017┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
Я2П
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_120037
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_120071┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
▓
╗non_trainable_variables
╝layers
йmetrics
 Йlayer_regularization_losses
┐layer_metrics
g	variables
htrainable_variables
iregularization_losses
k__call__
*l&call_and_return_all_conditional_losses
&l"call_and_return_conditional_losses"
_generic_user_object
п2Н
.__inference_leaky_re_lu_2_layer_call_fn_120076б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
з2­
I__inference_leaky_re_lu_2_layer_call_and_return_conditional_losses_120081б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
▓
└non_trainable_variables
┴layers
┬metrics
 ├layer_regularization_losses
─layer_metrics
n	variables
otrainable_variables
pregularization_losses
r__call__
*s&call_and_return_all_conditional_losses
&s"call_and_return_conditional_losses"
_generic_user_object
м2¤
(__inference_flatten_layer_call_fn_120086б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ь2Ж
C__inference_flatten_layer_call_and_return_conditional_losses_120092б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
$:"	ђ2z_log_mean/kernel
:2z_log_mean/bias
 "
trackable_dict_wrapper
.
t0
u1"
trackable_list_wrapper
.
t0
u1"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
┼non_trainable_variables
кlayers
Кmetrics
 ╚layer_regularization_losses
╔layer_metrics
w	variables
xtrainable_variables
yregularization_losses
{__call__
*|&call_and_return_all_conditional_losses
&|"call_and_return_conditional_losses"
_generic_user_object
Н2м
+__inference_z_log_mean_layer_call_fn_120101б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
­2ь
F__inference_z_log_mean_layer_call_and_return_conditional_losses_120111б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
#:!	ђ2z_log_var/kernel
:2z_log_var/bias
 "
trackable_dict_wrapper
.
}0
~1"
trackable_list_wrapper
.
}0
~1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
╩non_trainable_variables
╦layers
╠metrics
 ═layer_regularization_losses
╬layer_metrics
ђ	variables
Ђtrainable_variables
ѓregularization_losses
ё__call__
+Ё&call_and_return_all_conditional_losses
'Ё"call_and_return_conditional_losses"
_generic_user_object
н2Л
*__inference_z_log_var_layer_call_fn_120120б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
№2В
E__inference_z_log_var_layer_call_and_return_conditional_losses_120130б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
¤non_trainable_variables
лlayers
Лmetrics
 мlayer_regularization_losses
Мlayer_metrics
Є	variables
ѕtrainable_variables
Ѕregularization_losses
І__call__
+ї&call_and_return_all_conditional_losses
'ї"call_and_return_conditional_losses"
_generic_user_object
╔2к
)__inference_sampling_layer_call_fn_114199ў
Љ▓Ї
FullArgSpec
argsџ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
С2р
D__inference_sampling_layer_call_and_return_conditional_losses_114491ў
Љ▓Ї
FullArgSpec
argsџ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
╦B╚
$__inference_signature_wrapper_119739input_1"ћ
Ї▓Ѕ
FullArgSpec
argsџ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
J
%0
&1
A2
B3
]4
^5"
trackable_list_wrapper
є
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
13"
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
.
%0
&1"
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
.
A0
B1"
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
.
]0
^1"
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
trackable_dict_wrapperЉ
!__inference__wrapped_model_118445в&#%$56B?A@QR^[]\tu}~4б1
*б'
%і"
input_1         
ф "џфќ
.
sampling"і
sampling         
2

z_log_mean$і!

z_log_mean         
0
	z_log_var#і 
	z_log_var         Л
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_119923|B?A@@б=
6б3
-і*
inputs                  @
p 
ф "2б/
(і%
0                  @
џ Л
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_119957|AB?@@б=
6б3
-і*
inputs                  @
p
ф "2б/
(і%
0                  @
џ Е
6__inference_batch_normalization_1_layer_call_fn_119890oB?A@@б=
6б3
-і*
inputs                  @
p 
ф "%і"                  @Е
6__inference_batch_normalization_1_layer_call_fn_119903oAB?@@б=
6б3
-і*
inputs                  @
p
ф "%і"                  @М
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_120037~^[]\Aб>
7б4
.і+
inputs                  ђ
p 
ф "3б0
)і&
0                  ђ
џ М
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_120071~]^[\Aб>
7б4
.і+
inputs                  ђ
p
ф "3б0
)і&
0                  ђ
џ Ф
6__inference_batch_normalization_2_layer_call_fn_120004q^[]\Aб>
7б4
.і+
inputs                  ђ
p 
ф "&і#                  ђФ
6__inference_batch_normalization_2_layer_call_fn_120017q]^[\Aб>
7б4
.і+
inputs                  ђ
p
ф "&і#                  ђ¤
O__inference_batch_normalization_layer_call_and_return_conditional_losses_119809|&#%$@б=
6б3
-і*
inputs                   
p 
ф "2б/
(і%
0                   
џ ¤
O__inference_batch_normalization_layer_call_and_return_conditional_losses_119843|%&#$@б=
6б3
-і*
inputs                   
p
ф "2б/
(і%
0                   
џ Д
4__inference_batch_normalization_layer_call_fn_119776o&#%$@б=
6б3
-і*
inputs                   
p 
ф "%і"                   Д
4__inference_batch_normalization_layer_call_fn_119789o%&#$@б=
6б3
-і*
inputs                   
p
ф "%і"                   г
D__inference_conv1d_1_layer_call_and_return_conditional_losses_119877d563б0
)б&
$і!
inputs          
ф ")б&
і
0         @
џ ё
)__inference_conv1d_1_layer_call_fn_119862W563б0
)б&
$і!
inputs          
ф "і         @Г
D__inference_conv1d_2_layer_call_and_return_conditional_losses_119991eQR3б0
)б&
$і!
inputs         @
ф "*б'
 і
0         ђ
џ Ё
)__inference_conv1d_2_layer_call_fn_119976XQR3б0
)б&
$і!
inputs         @
ф "і         ђф
B__inference_conv1d_layer_call_and_return_conditional_losses_119763d3б0
)б&
$і!
inputs         
ф ")б&
і
0          
џ ѓ
'__inference_conv1d_layer_call_fn_119748W3б0
)б&
$і!
inputs         
ф "і          і
C__inference_encoder_layer_call_and_return_conditional_losses_119263┬&#%$56B?A@QR^[]\tu}~<б9
2б/
%і"
input_1         
p 

 
ф "jбg
`џ]
і
0/0         
і
0/1         
і
0/2         
џ і
C__inference_encoder_layer_call_and_return_conditional_losses_119326┬%&#$56AB?@QR]^[\tu}~<б9
2б/
%і"
input_1         
p

 
ф "jбg
`џ]
і
0/0         
і
0/1         
і
0/2         
џ Ѕ
C__inference_encoder_layer_call_and_return_conditional_losses_119537┴&#%$56B?A@QR^[]\tu}~;б8
1б.
$і!
inputs         
p 

 
ф "jбg
`џ]
і
0/0         
і
0/1         
і
0/2         
џ Ѕ
C__inference_encoder_layer_call_and_return_conditional_losses_119684┴%&#$56AB?@QR]^[\tu}~;б8
1б.
$і!
inputs         
p

 
ф "jбg
`џ]
і
0/0         
і
0/1         
і
0/2         
џ ▀
(__inference_encoder_layer_call_fn_118904▓&#%$56B?A@QR^[]\tu}~<б9
2б/
%і"
input_1         
p 

 
ф "ZџW
і
0         
і
1         
і
2         ▀
(__inference_encoder_layer_call_fn_119200▓%&#$56AB?@QR]^[\tu}~<б9
2б/
%і"
input_1         
p

 
ф "ZџW
і
0         
і
1         
і
2         я
(__inference_encoder_layer_call_fn_119379▒&#%$56B?A@QR^[]\tu}~;б8
1б.
$і!
inputs         
p 

 
ф "ZџW
і
0         
і
1         
і
2         я
(__inference_encoder_layer_call_fn_119432▒%&#$56AB?@QR]^[\tu}~;б8
1б.
$і!
inputs         
p

 
ф "ZџW
і
0         
і
1         
і
2         Ц
C__inference_flatten_layer_call_and_return_conditional_losses_120092^4б1
*б'
%і"
inputs         ђ
ф "&б#
і
0         ђ
џ }
(__inference_flatten_layer_call_fn_120086Q4б1
*б'
%і"
inputs         ђ
ф "і         ђГ
I__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_119967`3б0
)б&
$і!
inputs         @
ф ")б&
і
0         @
џ Ё
.__inference_leaky_re_lu_1_layer_call_fn_119962S3б0
)б&
$і!
inputs         @
ф "і         @»
I__inference_leaky_re_lu_2_layer_call_and_return_conditional_losses_120081b4б1
*б'
%і"
inputs         ђ
ф "*б'
 і
0         ђ
џ Є
.__inference_leaky_re_lu_2_layer_call_fn_120076U4б1
*б'
%і"
inputs         ђ
ф "і         ђФ
G__inference_leaky_re_lu_layer_call_and_return_conditional_losses_119853`3б0
)б&
$і!
inputs          
ф ")б&
і
0          
џ Ѓ
,__inference_leaky_re_lu_layer_call_fn_119848S3б0
)б&
$і!
inputs          
ф "і          ╠
D__inference_sampling_layer_call_and_return_conditional_losses_114491ЃZбW
PбM
KџH
"і
inputs/0         
"і
inputs/1         
ф "%б"
і
0         
џ Б
)__inference_sampling_layer_call_fn_114199vZбW
PбM
KџH
"і
inputs/0         
"і
inputs/1         
ф "і         Ъ
$__inference_signature_wrapper_119739Ш&#%$56B?A@QR^[]\tu}~?б<
б 
5ф2
0
input_1%і"
input_1         "џфќ
.
sampling"і
sampling         
2

z_log_mean$і!

z_log_mean         
0
	z_log_var#і 
	z_log_var         Д
F__inference_z_log_mean_layer_call_and_return_conditional_losses_120111]tu0б-
&б#
!і
inputs         ђ
ф "%б"
і
0         
џ 
+__inference_z_log_mean_layer_call_fn_120101Ptu0б-
&б#
!і
inputs         ђ
ф "і         д
E__inference_z_log_var_layer_call_and_return_conditional_losses_120130]}~0б-
&б#
!і
inputs         ђ
ф "%б"
і
0         
џ ~
*__inference_z_log_var_layer_call_fn_120120P}~0б-
&б#
!і
inputs         ђ
ф "і         