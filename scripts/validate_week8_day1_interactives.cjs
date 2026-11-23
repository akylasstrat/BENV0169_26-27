/* Run with Node.js, or load in the Codex Node REPL. No browser or network required. */
'use strict';
const fs=require('node:fs'),path=require('node:path'),vm=require('node:vm'),assert=require('node:assert/strict');
const root=path.resolve(__dirname,'..'),folder=path.join(root,'teaching'),output=path.join(folder,'week8_day1_outputs');
const temporary=path.join(root,'tmp/week8_html_20261119');fs.mkdirSync(temporary,{recursive:true});
const read=name=>fs.readFileSync(path.join(folder,name),'utf8');
function scripts(html){return [...html.matchAll(/<script([^>]*)>([\s\S]*?)<\/script>/g)].filter(m=>!m[1].includes('application/json')).map(m=>m[2]);}
function payload(html,id){return JSON.parse(html.match(new RegExp('<script id="'+id+'" type="application/json">([\\s\\S]*?)<\\/script>'))[1]);}
const near=(x,y,tolerance=1e-7)=>assert.ok(Math.abs(x-y)<=tolerance,`${x} differs from ${y}`);

// A small isolated DOM double exercises the actual inline event handlers and SVG output.
class Element{
  constructor(tag,attributes={}){this.tag=tag;this.attributes={...attributes};this.children=[];this.value=attributes.value||'';this.checked='checked' in attributes;this.hidden='hidden' in attributes;this.disabled=false;this.textContent='';this.listeners={};this.classes=new Set();this.classList={toggle:(name,on)=>on?this.classes.add(name):this.classes.delete(name)};}
  appendChild(e){this.children.push(e);return e;}
  replaceChildren(...children){this.children=children;}
  setAttribute(key,value){this.attributes[key]=String(value);}
  getAttribute(key){return this.attributes[key];}
  addEventListener(type,handler){(this.listeners[type]??=[]).push(handler);}
  dispatch(type,event={}){if(type==='click'&&this.disabled)throw Error('Clicked disabled control');this['on'+type]?.(event);for(const handler of this.listeners[type]||[])handler(event);}
  focus(){}
  setPointerCapture(){}
  createSVGPoint(){return {x:0,y:0,matrixTransform(){return {x:this.x,y:this.y};}};}
  getScreenCTM(){return {inverse(){return {};}};}
}
function makeDom(html){
  const nodes={};
  for(const match of html.matchAll(/<([a-z][\w:-]*)\b([^>]*)>/gi)){
    const attrs={};for(const a of match[2].matchAll(/([\w:-]+)(?:="([^"]*)")?/g))attrs[a[1]]=a[2]??'';
    if(attrs.id){assert.ok(!nodes[attrs.id],'Duplicate id: '+attrs.id);nodes[attrs.id]=new Element(match[1],attrs);}
  }
  for(const match of html.matchAll(/<script id="([^"]+)" type="application\/json">([\s\S]*?)<\/script>/g))nodes[match[1]].textContent=match[2];
  if(nodes.mode)nodes.mode.value='rolling';
  if(nodes.steps)nodes.steps.children=Array.from({length:6},()=>new Element('li'));
  return {nodes,getElementById(id){assert.ok(nodes[id],'Missing element: '+id);return nodes[id];},createElementNS(ns,tag){return new Element(tag);}};
}
function load(html){
  const document=makeDom(html),context=vm.createContext({document,console,requestAnimationFrame:fn=>fn()});
  for(const script of scripts(html))new vm.Script(script).runInContext(context);
  return {context,nodes:document.nodes,document};
}
function xml(e){
  const escape=s=>String(s).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('"','&quot;');
  const attributes={...e.attributes};if(e.tag==='svg'){attributes.xmlns='http://www.w3.org/2000/svg';attributes['font-family']='Arial, sans-serif';}
  return '<'+e.tag+' '+Object.entries(attributes).map(([k,v])=>k+'="'+escape(v)+'"').join(' ')+'>'+escape(e.textContent)+e.children.map(xml).join('')+'</'+e.tag+'>';
}
function saveSvg(element,name){fs.writeFileSync(path.join(temporary,name+'.svg'),xml(element));}
function update(ui,id,value,event='input'){ui.nodes[id].value=String(value);ui.nodes[id].dispatch(event);}

const heatingHtml=read('tutorial_8_1_two_interval_heating.html'),playerHtml=read('tutorial_8_1_mpc_player.html');
for(const html of [heatingHtml,playerHtml]){
  assert.ok(!/<script[^>]+src=|<link[^>]+href=/i.test(html),'External dependency');
  assert.ok(!/\b(fetch|XMLHttpRequest|WebSocket)\s*\(/.test(html),'Network request');
  for(const script of scripts(html))new vm.Script(script);
}

const heating=load(heatingHtml),cases=JSON.parse(fs.readFileSync(path.join(output,'interactive_lp_oracles.json'),'utf8'));
let worstGap=0,feasibleCases=0;
for(const test of cases){
  const solution=heating.context.solveTwoInterval(test.input);
  assert.equal(solution.value!==null,test.feasible,'Feasibility differs from independent HiGHS');
  if(test.feasible){
    feasibleCases++;near(solution.value,test.value,1e-7);worstGap=Math.max(worstGap,Math.abs(solution.value-test.value));
    for(const p of solution.points){assert.ok(heating.context.feasible(p,solution.hard));near(heating.context.evaluate(test.input,p).value,solution.value);}
  }
}
assert.equal(heating.nodes.feasibility.textContent,'Feasible schedule');
update(heating,'candidate0',2.5);assert.equal(heating.nodes['point-p0'].textContent,'2.50');
heating.nodes.reveal.checked=true;heating.nodes.reveal.dispatch('input');assert.equal(heating.nodes.optimal.hidden,false);assert.ok(heating.nodes.optimal.textContent.includes('J*='));
saveSvg(heating.nodes.graph,'heating-default');saveSvg(heating.nodes.trajectory,'heating-trajectory');
heating.nodes['shortfall-preset'].dispatch('click');assert.equal(heating.nodes.feasibility.textContent,'Feasible schedule');assert.ok(heating.nodes['constraint-status'].textContent.includes('penalised'));
heating.nodes.reveal.checked=true;heating.nodes.reveal.dispatch('input');saveSvg(heating.nodes.graph,'heating-shortfall');
update(heating,'initial',24.5);assert.equal(heating.nodes.feasibility.textContent,'Hard limit violated');assert.ok(heating.nodes.optimal.textContent.includes('exclude every schedule'));
heating.nodes.reset.dispatch('click');heating.nodes.graph.dispatch('keydown',{key:'ArrowRight',shiftKey:false,preventDefault(){}});assert.equal(heating.nodes['point-p0'].textContent,'1.10');
heating.nodes.graph.dispatch('pointerdown',{button:0,pointerId:1,clientX:59+548*.6,clientY:23+548*.6});near(Number(heating.nodes['point-p0'].textContent),3,.01);near(Number(heating.nodes['point-p1'].textContent),2,.01);heating.nodes.graph.dispatch('pointerup');
heating.nodes['upper-preset'].dispatch('click');heating.nodes.reveal.checked=true;heating.nodes.reveal.dispatch('input');saveSvg(heating.nodes.graph,'heating-upper-bound');
// Every shaded tile is clipped to the hard feasible set, including binding bounds.
const upperSolution=heating.context.solveTwoInterval(heating.context.model());
for(const tile of heating.nodes.graph.children.filter(e=>e.tag==='polygon'&&String(e.attributes.fill).startsWith('rgb('))){
  for(const position of tile.attributes.points.split(' ')){
    const [x,y]=position.split(',').map(Number),p={x:(x-59)/548*5,y:(23+548-y)/548*5};
    assert.ok(heating.context.feasible(p,upperSolution.hard),'Shading extends beyond a hard limit');
  }
}

const player=load(playerHtml),data=payload(playerHtml,'experiment-data');
assert.equal(player.nodes.optimise.disabled,false);assert.equal(player.nodes.apply.disabled,true);
player.nodes.optimise.dispatch('click');assert.equal(player.nodes.apply.disabled,false);assert.ok(player.nodes.action.textContent.includes(data.runs.rolling.plans[0].power[0].toFixed(3)));
const pastLine=()=>player.nodes.temperature.children.find(e=>e.tag==='polyline'&&e.attributes.stroke==='#192c3c');
assert.equal(pastLine().attributes.points.split(' ').length,1);
player.nodes.apply.dispatch('click');assert.equal(pastLine().attributes.points.split(' ').length,1);assert.equal(player.nodes.observe.disabled,false);
player.nodes.observe.dispatch('click');assert.equal(pastLine().attributes.points.split(' ').length,2);assert.ok(player.nodes.answer.textContent.includes(data.runs.rolling.measured[1].toFixed(3)));
player.nodes.repeat.dispatch('click');assert.equal(player.nodes.clock.textContent,'00:30');assert.equal(player.nodes.horizon.textContent,'48 × 30 min');
player.nodes.optimise.dispatch('click');saveSvg(player.nodes.temperature,'player-second-temperature');saveSvg(player.nodes.power,'player-second-power');
update(player,'origin',24);player.nodes.optimise.dispatch('click');saveSvg(player.nodes.temperature,'player-noon-temperature');saveSvg(player.nodes.power,'player-noon-power');
player.nodes.previous.checked=false;player.nodes.previous.dispatch('change');assert.ok(!player.nodes.temperature.children.some(e=>e.attributes.stroke==='#a6b0b8'));
player.nodes.truth.checked=true;player.nodes.truth.dispatch('change');const truth=player.nodes.temperature.children.find(e=>e.attributes.stroke==='#287a53');assert.equal(truth.attributes.points.split(' ').length,25);
update(player,'mode','shrinking','change');update(player,'origin',47);assert.equal(player.nodes.horizon.textContent,'1 × 30 min');assert.equal(player.nodes.endpoint.textContent,'4 Feb 00:00');assert.equal(player.nodes.forecast.textContent,'Final interval');
for(const id of ['optimise','apply','observe'])player.nodes[id].dispatch('click');assert.equal(player.nodes.repeat.disabled,true);assert.ok(player.nodes.status.textContent.includes('48 actions and 49 state boundaries'));
saveSvg(player.nodes.temperature,'player-final-temperature');saveSvg(player.nodes.power,'player-final-power');
player.nodes.reset.dispatch('click');assert.equal(player.nodes.clock.textContent,'00:00');assert.equal(player.nodes.truth.checked,false);

const report=JSON.parse(fs.readFileSync(path.join(output,'interactives_validation.json'),'utf8'));
report.javascript={status:'PASS',lp_cases:cases.length,feasible_cases:feasibleCases,max_lp_objective_difference_gbp:worstGap,
  checks:['full inline JavaScript executes','all queried elements exist','slider and preset handlers','point dragging and keyboard controls',
          'soft shortfalls remain feasible','infeasible upper limits','optimum reveal','four replay stages','past observations remain hidden until reveal',
          'rolling and shrinking modes','origin seeking','previous-plan and truth toggles','final interval','restart','offline dependencies'],
  ui_method:'Isolated DOM and SVG execution; live Chrome preview was unavailable.'};
fs.writeFileSync(path.join(output,'interactives_validation.json'),JSON.stringify(report,null,2));
console.log('PASS:',cases.length,'live JavaScript optima checked against HiGHS; both page controllers and SVG renders exercised.');
