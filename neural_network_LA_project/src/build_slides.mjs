// Author the editable PPTX from the same recorded results as the writeup.
// Requires the Codex bundled artifact tool, Python and presentation skill.
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const modules = process.env.MATHS_NODE_MODULES;
const skill = process.env.MATHS_PRESENTATION_SKILL;
const python = process.env.MATHS_PYTHON;
if (!modules || !skill || !python) throw new Error('Set MATHS_NODE_MODULES, MATHS_PRESENTATION_SKILL and MATHS_PYTHON to the bundled runtime paths.');
process.env.RUNTIME_NODE_MODULES = modules;
const { Presentation, PresentationFile } = await import(pathToFileURL(path.join(modules, '@oai/artifact-tool/dist/artifact_tool.mjs')).href);
const { finalizePresentation, applyPresentationChartFont } = await import(pathToFileURL(path.join(skill, 'container_tools/artifact_tool_utils.mjs')).href);
const build = path.join(root, '.build');
await fs.mkdir(build, { recursive: true });
try { await fs.symlink(modules, path.join(build, 'node_modules')); } catch (e) { if (e.code !== 'EEXIST') throw e; }
const content = JSON.parse(await fs.readFile(path.join(root, 'presentation/slides.json'), 'utf8'));
const presentation = Presentation.create({ slideSize: { width: 1280, height: 720 } });
const font = 'Arial';
const ink = '#183c2d';
const paper = '#f2f1e8';

function text(slide, value, x, y, w, h, size, color=ink, bold=false) {
  const box = slide.shapes.add({ geometry:'textbox', position:{left:x,top:y,width:w,height:h},
    fill:'none', line:{fill:'none',width:0} });
  box.text = value;
  box.text.style = {typeface:font, fontSize:size, color, bold};
  return box;
}
function table(slide, values) {
  const rows=values.length, columns=values[0].length;
  const t=slide.tables.add({rows,columns,left:64,top:190,width:1152,height:rows*64,values});
  t.borders.assign({fill:'#b6c6b4',width:0.6,style:'solid'});
  for(let r=0;r<rows;r++) {
    t.rows[r].height=64;
    for(let c=0;c<columns;c++) {
      const cell=t.getCell(r,c);
      cell.fill=r===0?'#dce8dc':paper;
      cell.text.style={typeface:font,fontSize:columns>5?22:25,bold:r===0,color:ink};
    }
  }
}
async function scientificImage(slide, filename) {
  const buffer = await fs.readFile(path.join(root, 'results/figures',filename));
  slide.images.add({ blob:buffer.buffer.slice(buffer.byteOffset,buffer.byteOffset+buffer.byteLength),
    contentType:'image/png',alt:filename,fit:'contain',
    position:{left:64,top:174,width:1152,height:355} });
}

for(const [i,c] of content.entries()) {
  const slide=presentation.slides.add();
  const cover=i===0;
  slide.background.fill=cover?ink:paper;
  text(slide,c.title,64,cover?160:58,1152,cover?180:105,cover?58:43,cover?paper:ink,true);
  if(c.table) {
    table(slide,c.table);
    if(c.body) text(slide,c.body,68,590,1136,55,26);
  } else if(i===8) {
    // Native PowerPoint chart retains editable epoch/loss evidence.
    const lines=(await fs.readFile(path.join(root,'results/history_deep_42.csv'),'utf8')).trim().split(/\r?\n/);
    // Excel chart workbooks retain 15 significant digits. Ten decimal places
    // preserve plotted loss precision while CSV histories retain full values.
    const rows=lines.slice(1).map(line=>line.split(',').map(v=>Number(Number(v).toFixed(10))));
    const chart=slide.charts.add('line',{
      position:{left:64,top:174,width:760,height:390},
      title:'Mean cross-entropy by completed updates',
      titleTextStyle:{fontSize:24},
      categories:rows.map(r=>String(r[0])),
      series:[{name:'Training',values:rows.map(r=>r[1]),line:{fill:'#3157b7',width:2,style:'solid'}},
              {name:'Validation',values:rows.map(r=>r[3]),line:{fill:'#d05b50',width:2,style:'solid'}}],
      hasLegend:true,legend:{position:'bottom',textStyle:{fontSize:20}},
      lineOptions:{smooth:false},
      xAxis:{textStyle:{fontSize:16},title:{text:'Completed updates',textStyle:{fontSize:18}}},
      yAxis:{numberFormatCode:'0.0',textStyle:{fontSize:18},title:{text:'Cross-entropy',textStyle:{fontSize:18}}},
      chartFill:paper,plotAreaFill:paper
    });
    applyPresentationChartFont(chart,{fontFamily:font});
    text(slide,c.body,870,200,330,340,28);
  } else if(c.image) {
    await scientificImage(slide,c.image);
    if(c.body) text(slide,c.body,68,555,1136,105,27);
  } else if(c.body) {
    text(slide,c.body,cover?68:90,cover?385:210,cover?1100:1100,cover?160:370,cover?32:35,cover?paper:ink);
  }
  text(slide,`${String(i+1).padStart(2,'0')} / ${content.length}`,1160,679,90,22,16,cover?'#a9c3ad':'#526a5c');
  slide.speakerNotes.text=c.notes;
  const preview=await presentation.export({slide,format:'png',scale:1});
  await fs.writeFile(path.join(build,`slide-${String(i+1).padStart(2,'0')}.png`),new Uint8Array(await preview.arrayBuffer()));
}
const candidatePath=path.join(build,'candidate.pptx');
await (await PresentationFile.exportPptx(presentation)).save(candidatePath);
const finalPath=path.join(root,'presentation/Neural_Network_Linear_Algebra.pptx');
await finalizePresentation({
  workspaceDir:root,candidatePath,finalPath,pythonExecutable:python,
  integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
  layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
  layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit',
    ...[5,10,11,14,15].flatMap(number=>['--require-native-table-slide',String(number)])],
  explicitTotalSlideCount:16,
  requiredNativeTableOwnerSlides:[5,10,11,14,15],
  requiredNativeChartOwnerSlides:[9],
  materializeLiteralChartWorkbooks:true,
  fontPolicy:{basis:'design',families:[font]},
  verifyArtifactToolImport:true,
  receiptPath:path.join(build,'deck.validation.json')
});
console.log(`Exported ${content.length} slides: ${finalPath}`);
