function [residual, T_order, T] = dynamic_resid(y, x, params, steady_state, T_order, T)
if nargin < 6
    T_order = -1;
    T = NaN(19, 1);
end
[T_order, T] = Smets_Wouters_2007.sparse.dynamic_resid_tt(y, x, params, steady_state, T_order, T);
residual = NaN(40, 1);
    residual(1) = (y(72)) - (params(9)*y(51)+(1-params(9))*y(58));
    residual(2) = (y(50)) - (y(51)*T(3));
    residual(3) = (y(51)) - (y(58)+y(57)-y(52));
    residual(4) = (y(52)) - (y(50)+y(39));
    residual(5) = (y(55)) - (y(75)+T(15)*(y(53)*1/T(5)+y(15)+y(95)*T(14)));
    residual(6) = (y(53)) - (y(73)*1/T(7)-y(59)+y(91)*(T(11)-(1-params(12)))/T(11)+y(93)*(1-params(12))/T(11));
    residual(7) = (y(54)) - (y(73)+y(14)*T(6)/(1+T(6))+y(94)*1/(1+T(6))+(y(57)-y(97))*T(16)-y(59)*T(7));
    residual(8) = (y(56)) - (y(74)+y(54)*(1-params(39)-T(1)*T(12)*T(13))+y(55)*T(1)*T(12)*T(13)+y(50)*(T(11)-(1-params(12)))*T(13));
    residual(9) = (y(56)) - (params(17)*(y(72)+params(9)*y(52)+(1-params(9))*y(57)));
    residual(10) = (y(58)) - (y(57)*params(22)+y(54)*T(8)-y(14)*T(9));
    residual(11) = (y(79)) - (y(39)*(1-T(12))+y(55)*T(12)+y(75)*T(5)*T(12));
    residual(12) = (y(60)) - (params(9)*y(62)+(1-params(9))*y(70)-y(72));
    residual(13) = (y(61)) - (T(3)*y(62));
    residual(14) = (y(62)) - (y(70)+y(68)-y(63));
    residual(15) = (y(63)) - (y(61)+y(40));
    residual(16) = (y(66)) - (y(75)+T(15)*(y(64)*1/T(5)+y(26)+y(106)*T(14)));
    residual(17) = (y(64)) - (y(109)-y(71)+y(73)*1/T(7)+y(102)*(T(11)-(1-params(12)))/T(11)+y(104)*(1-params(12))/T(11));
    residual(18) = (y(65)) - (y(73)+y(25)*T(6)/(1+T(6))+y(105)*1/(1+T(6))+(y(68)-y(108))*T(16)-(y(71)-y(109))*T(7));
    residual(19) = (y(67)) - (y(74)+y(65)*(1-params(39)-T(1)*T(12)*T(13))+y(66)*T(1)*T(12)*T(13)+y(61)*(T(11)-(1-params(12)))*T(13));
    residual(20) = (y(67)) - (params(17)*(y(72)+params(9)*y(63)+(1-params(9))*y(68)));
    residual(21) = (y(69)) - (y(77)+T(17)*(params(20)*y(29)+y(109)*T(14)+y(60)*T(18)));
    residual(22) = (y(70)) - (y(78)+y(30)*T(15)+y(110)*T(14)/(1+T(14))+y(29)*params(18)/(1+T(14))-y(69)*(1+params(18)*T(14))/(1+T(14))+y(109)*T(14)/(1+T(14))+(params(22)*y(68)+y(65)*T(8)-y(25)*T(9)-y(70))*T(19));
    residual(23) = (y(71)) - (y(69)*params(25)*(1-params(28))+(1-params(28))*params(27)*(y(67)-y(56))+params(26)*(y(67)-y(56)-y(27)+y(16))+params(28)*y(31)+y(76));
    residual(24) = (y(72)) - (params(29)*y(32)+x(1));
    residual(25) = (y(73)) - (params(31)*y(33)+x(2));
    residual(26) = (y(74)) - (params(32)*y(34)+x(3)+x(1)*params(2));
    residual(27) = (y(75)) - (params(34)*y(35)+x(4));
    residual(28) = (y(76)) - (params(35)*y(36)+x(5));
    residual(29) = (y(77)) - (params(36)*y(37)+y(49)-params(8)*y(9));
    residual(30) = (y(49)) - (x(6));
    residual(31) = (y(78)) - (params(37)*y(38)+y(48)-params(7)*y(8));
    residual(32) = (y(48)) - (x(7));
    residual(33) = (y(80)) - (y(40)*(1-T(12))+y(66)*T(12)+y(75)*params(11)*T(4)*T(12));
    residual(34) = (y(44)) - (params(38)+y(67)-y(27));
    residual(35) = (y(45)) - (params(38)+y(65)-y(25));
    residual(36) = (y(46)) - (params(38)+y(66)-y(26));
    residual(37) = (y(47)) - (params(38)+y(70)-y(30));
    residual(38) = (y(43)) - (params(5)+y(69));
    residual(39) = (y(42)) - (y(71)+100*((1+params(5)/100)/T(10)-1));
    residual(40) = (y(41)) - (y(68)+params(4));
end
