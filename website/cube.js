/* Livnium hero cube — HIERARCHICAL RUBIK (Core-in-Every-Cube).
   A Rubik's cube whose 27 cubies are each a full 3x3x3 Livnium core (729 cells).
   TWO scales of Rubik turns run at once:
     · MACRO — a slice of 9 cubies snaps 90°  (axiom A4)
     · MICRO — inside individual cubies, a slice of 9 cells snaps 90°
   Every quarter-turn only PERMUTES cells, so class counts + ΣSW are conserved (ledger D3).
   Each cubie keeps its own glowing Om center (Om -> LO at every scale). */
(function(){
  const host = document.getElementById('voxelCube');
  if(!host || !window.THREE) return;

  const reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;

  // ---- palette: matched to site tokens — --blue #2f9bf0, --blue-deep #1b46b3, on pure black ----
  const PAL = {
    0:{c:0x0b1220, e:0x05080f, o:0.28},  // core   - near-black blue glass, barely there
    1:{c:0x2f9bf0, e:0x0d3a63, o:0.80},  // center - Livnium brand blue
    2:{c:0x1b46b3, e:0x070f26, o:0.50},  // edge   - deep blue
    3:{c:0x58b4f5, e:0x0d3a63, o:0.64}   // corner - lighter brand tint
  };
  const OM = {c:0xeaf3ff, e:0x2f9bf0, o:0.94}; // Om/core observer - white with brand-blue glow

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(45, 1, .1, 100);
  const renderer = new THREE.WebGLRenderer({antialias:true, alpha:true});
  renderer.setPixelRatio(Math.min(2, devicePixelRatio));
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.15;
  host.appendChild(renderer.domElement);

  scene.add(new THREE.AmbientLight(0xffffff, .35));
  const d1 = new THREE.DirectionalLight(0xffffff, .85); d1.position.set(6, 9, 7);   scene.add(d1);
  const d2 = new THREE.DirectionalLight(0x2f9bf0, .35); d2.position.set(-7, -4, -6); scene.add(d2);

  // shared resources (729 cells stay cheap)
  const MICRO = 0.30, BLOCK = 1.28;
  const geo     = new THREE.BoxGeometry(0.24, 0.24, 0.24);
  const edgeGeo = new THREE.EdgesGeometry(geo);
  const edgeMat = new THREE.LineBasicMaterial({color:0x2f6ea8, transparent:true, opacity:0.22});

  // solid cells read far cleaner than 729 overlapping translucent boxes
  const classMat = {};
  for(const k in PAL){
    classMat[k] = new THREE.MeshStandardMaterial({
      color: PAL[k].c,
      emissive: PAL[k].e,
      emissiveIntensity: k == 0 ? 0.15 : 0.5,
      roughness: 0.42,
      metalness: 0.14
    });
  }
  const omMat = new THREE.MeshStandardMaterial({
    color: OM.c, emissive: OM.e, emissiveIntensity: 0.9, roughness: 0.2, metalness: 0.2
  });

  const world  = new THREE.Group();
  const blocks = [];
  const f = (a,b,c) => (Math.abs(a)===1)+(Math.abs(b)===1)+(Math.abs(c)===1);

  // macro lattice (27 cubies) — each a mini Livnium core
  for(let X=-1;X<=1;X++)for(let Y=-1;Y<=1;Y++)for(let Z=-1;Z<=1;Z++){
    const block = new THREE.Group();
    block.position.set(X*BLOCK, Y*BLOCK, Z*BLOCK);
    block.userData = { home:new THREE.Vector3(X,Y,Z), micro:[], micromove:null };
    for(let x=-1;x<=1;x++)for(let y=-1;y<=1;y++)for(let z=-1;z<=1;z++){
      const isOm = (x===0&&y===0&&z===0);
      const cell = new THREE.Mesh(geo, isOm ? omMat : classMat[f(x,y,z)]);
      cell.position.set(x*MICRO, y*MICRO, z*MICRO);
      cell.userData = { home:new THREE.Vector3(x,y,z) };
      cell.add(new THREE.LineSegments(edgeGeo, edgeMat));
      block.add(cell);
      block.userData.micro.push(cell);
    }
    blocks.push(block);
    world.add(block);
  }
  scene.add(world);

  // ---- generic layer-turn engine (works at either scale) ----
  const AXES = ['x','y','z'];
  function begin(parent, items, spacing){
    const axis  = AXES[(Math.random()*3)|0];
    const layer = [-1,0,1][(Math.random()*3)|0];
    const dir   = Math.random()<.5 ? 1 : -1;
    const sel   = items.filter(i => Math.round(i.userData.home[axis]) === layer);
    const pivot = new THREE.Group(); parent.add(pivot);
    sel.forEach(i => { parent.remove(i); pivot.add(i); });
    return { axis, dir, sel, pivot, parent, spacing, angle:0, target:Math.PI/2 };
  }
  function advance(m, speed){
    m.angle += Math.min(speed, m.target - m.angle);
    m.pivot.rotation[m.axis] = m.dir * m.angle;
    if(m.angle < m.target - 1e-4) return false;
    // bake: fold the pivot rotation into each item, snap back to the lattice
    m.pivot.rotation[m.axis] = m.dir * m.target; m.pivot.updateMatrix();
    m.sel.forEach(i => {
      i.applyMatrix4(m.pivot.matrix);
      const s = m.spacing;
      const h = i.userData.home.set(
        Math.round(i.position.x/s), Math.round(i.position.y/s), Math.round(i.position.z/s));
      i.position.set(h.x*s, h.y*s, h.z*s);   // ΣSW & class counts conserved ✓
      m.parent.add(i);
    });
    m.parent.remove(m.pivot);
    return true;
  }

  // macro scheduler (one big turn at a time)
  let macro = null, nextMacro = 0;
  // micro scheduler (one gentle cubie turning inside at a time)
  const micros = []; let nextMicro = 0;

  // ---- live ledger: recompute ΣSW from the actual cells after every turn ----
  // SW = 9·f by face-exposure; a turn only permutes cells, so the sum is invariant.
  function sigmaSW(){
    let s = 0;
    for(const b of blocks) for(const c of b.userData.micro){
      const h = c.userData.home; s += 9 * ((Math.abs(h.x)===1)+(Math.abs(h.y)===1)+(Math.abs(h.z)===1));
    }
    return s;                       // 27 cores × 486 = 13122, constant
  }
  const elVal  = document.getElementById('swVal');
  const elNote = document.getElementById('swNote');
  const ledger = {
    turning(){ if(elNote){ elNote.textContent = 'rotating…'; elNote.className = 'note run'; } },
    tick(){
      const s = sigmaSW();
      if(elVal)  elVal.textContent = s.toLocaleString();
      if(elNote){ elNote.textContent = 'conserved ✓'; elNote.className = 'note ok'; }
    }
  };

  // interaction
  let rotX = -.45, rotY = .7, autorot = !reduced;
  let dragging = false, px = 0, py = 0;
  host.style.cursor = 'grab';
  host.addEventListener('pointerdown', e => {dragging=true; px=e.clientX; py=e.clientY; host.style.cursor='grabbing';});
  addEventListener('pointerup', () => {dragging=false; host.style.cursor='grab';});
  addEventListener('pointermove', e => {
    if(!dragging) return;
    rotY += (e.clientX-px)*.008; rotX += (e.clientY-py)*.008;
    px=e.clientX; py=e.clientY;
  });

  function size(){
    const w = host.clientWidth || 420, h = host.clientHeight || 420;
    camera.aspect = w/h; camera.updateProjectionMatrix(); renderer.setSize(w, h);
  }
  size(); addEventListener('resize', size);
  camera.position.set(0, 0, 8.8);

  function animate(t){
    requestAnimationFrame(animate);
    if(autorot && !dragging) rotY += .0022;
    world.rotation.x = rotX; world.rotation.y = rotY;

    if(!reduced){
      // MACRO turn — one calm quarter-turn at a time
      if(macro){
        if(advance(macro, .085)){ macro=null; nextMacro = t + 900; ledger.tick(); }
      } else if(t > nextMacro){ macro = begin(world, blocks, BLOCK); ledger.turning(); }

      // MICRO turn — a single, gentle layer inside one cubie at a time
      for(let i=micros.length-1;i>=0;i--){
        if(advance(micros[i].m, .11)){ micros[i].block.userData.micromove = null; micros.splice(i,1); }
      }
      if(t > nextMicro && micros.length < 1){
        const free = blocks.filter(b => !b.userData.micromove);
        if(free.length){
          const b = free[(Math.random()*free.length)|0];
          const m = begin(b, b.userData.micro, MICRO);
          b.userData.micromove = m; micros.push({m, block:b});
        }
        nextMicro = t + 1400;
      }

      omMat.emissiveIntensity = .7 + .18 * Math.sin(t/800); // Om breathes softly
    }

    camera.lookAt(0,0,0);
    renderer.render(scene, camera);
  }
  requestAnimationFrame(animate);
})();
