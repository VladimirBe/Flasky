//скрыть блок с данными о спине
function Hide()
{
	document.getElementById('spin_block').style.display = 'none';
}

//показать блок с данными о спине
function Show(num, value_no, value_yes, color)
{
	var str = "<p class='hide_link'><span class='pseudo_link' onclick='Hide();'>Скрыть</span></p><p>Выбранное поле: <b style='font-size:11px; border:1px solid #cccc00; padding:5px; color:#fff; background:" + color + "'>" + num + "</b></p>";
	str = str + "<p>Текущее количество невыпадений: <b>" + value_no + "</b></p><p>Выпадений за текущий месяц: <b>" + value_yes + "</b></p>";
	document.getElementById('spin_block').innerHTML = str;
	document.getElementById('spin_block').style.display = 'block';
}

